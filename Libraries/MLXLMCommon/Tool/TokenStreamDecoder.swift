// Copyright © 2026 Apple Inc.

/// A protocol-neutral semantic event decoded from a model's generated token stream.
package enum TokenStreamEvent: Sendable, Equatable {
    case reasoning(String)
    case response(String)
    case toolCall(ToolCall)
    /// A framed protocol rejected malformed output. Public generation logs it;
    /// package-level consumers can observe it and decide whether to retry.
    case protocolError(String)
    /// A tool-call-shaped model output rejected by parsing or authorization.
    case rejectedToolCall(RejectedToolCall)
    case stop
}

/// Decodes model-specific token streams into response and tool-call events.
///
/// The generation loop owns iteration and cancellation. A decoder owns the
/// response protocol: token framing, semantic stop tokens, and event routing.
/// This keeps model-specific protocols out of the generic evaluation loop.
package protocol TokenStreamDecoder {
    /// Semantic boundaries in addition to the model's ordinary EOS tokens.
    var additionalStopTokenIDs: Set<Int> { get }

    /// Whether semantic stop tokens must be passed through `push` before the
    /// generation loop terminates.
    var receivesStopTokens: Bool { get }

    /// Whether the decoder is currently consuming private reasoning payload.
    /// Sample this before feeding a token to attribute usage accurately.
    var isInsideReasoning: Bool { get }

    /// Tool-call-shaped outputs rejected so far during this generation.
    ///
    /// Decoders whose response protocol never rejects tool calls report zero.
    var rejectedToolCallCount: Int { get }

    /// Calls promoted by bounded cross-dialect recovery during this generation.
    var recoveredToolCallCount: Int { get }

    /// Consumes one generated token. Returns `false` when decoding should stop
    /// because of either a semantic boundary or consumer termination.
    mutating func push(_ token: Int, emit: (TokenStreamEvent) -> Bool) -> Bool

    /// Flushes any buffered events at the end of generation. Returns `false`
    /// when decoding stopped before all buffered events were delivered.
    mutating func finish(emit: (TokenStreamEvent) -> Bool) -> Bool
}

extension TokenStreamDecoder {
    package var additionalStopTokenIDs: Set<Int> { [] }
    package var receivesStopTokens: Bool { false }
    package var isInsideReasoning: Bool { false }
    package var rejectedToolCallCount: Int { 0 }
    package var recoveredToolCallCount: Int { 0 }
}

/// Decoder for ordinary detokenized tool-call syntaxes.
///
/// When the model declares a ``ReasoningConfig``, this also splits its thinking
/// spans out of the response before the tool-call parser sees them. Framed token
/// protocols (Harmony, Onyx) decode their own reasoning and never reach here.
struct StandardTokenStreamDecoder: TokenStreamDecoder {
    private let tokenizer: any Tokenizer
    private var detokenizer: any StreamingDetokenizer
    private let format: ToolCallFormat
    private let tools: [[String: any Sendable]]?
    private let toolCallPolicy: ToolCallPolicy
    private let toolCallProcessor: ToolCallProcessor
    private var stopStringFilter: StopStringFilter

    /// Absent for models with no reasoning protocol, which then decode and cost
    /// exactly what they did before.
    private var reasoningEmitter: ReasoningEventEmitter?

    /// Set when the reasoning delimiters are single tokens. They are then matched
    /// by id, and the emitter does not scan for their text. The same holds for each
    /// implicit end delimiter that is a single token.
    private let reasoningTokens: ReasoningDelimiterTokens?

    /// Buffered reasoning text from the first explicit tool-protocol opener,
    /// so a rehearsed call inside `{"arg": "<thinking>"}` is rejected or surfaced
    /// structurally instead of leaking as prose. Fork divergence (Phase 53):
    /// upstream routes reasoning straight to the stream.
    private var reasoningToolAttempt: String?

    init(
        tokenizer: any Tokenizer,
        format: ToolCallFormat,
        tools: [[String: any Sendable]]?,
        stopStrings: Set<String>,
        toolCallPolicy: ToolCallPolicy = .init(),
        reasoning: (config: ReasoningConfig, primedInside: Bool)? = nil
    ) {
        self.tokenizer = tokenizer
        self.format = format
        self.tools = tools
        self.toolCallPolicy = toolCallPolicy
        self.detokenizer = tokenizer.makeStreamingDetokenizer()
        self.toolCallProcessor = ToolCallProcessor(
            format: format, tools: tools, toolCallPolicy: toolCallPolicy)
        self.stopStringFilter = StopStringFilter(stopStrings: stopStrings)
        let reasoningTokens = reasoning.flatMap {
            ReasoningDelimiterTokens(config: $0.config, tokenizer: tokenizer)
        }
        self.reasoningTokens = reasoningTokens
        self.reasoningEmitter = reasoning.map {
            ReasoningEventEmitter(
                config: $0.config, primedInside: $0.primedInside,
                delimiterTokens: reasoningTokens)
        }
    }

    var rejectedToolCallCount: Int { toolCallProcessor.rejectedToolCallCount }
    var recoveredToolCallCount: Int { toolCallProcessor.recoveredToolCallCount }

    var isInsideReasoning: Bool { reasoningEmitter?.isInsideReasoning ?? false }

    mutating func push(_ token: Int, emit: (TokenStreamEvent) -> Bool) -> Bool {
        if let boundary = reasoningTokens?.boundary(for: token),
            reasoningEmitter?.isBoundary(boundary) == true
        {
            return push(boundary, emit: emit)
        }
        detokenizer.append(token: token)
        guard let chunk = detokenizer.next() else { return true }
        return process(chunk, emit: emit)
    }

    /// Handles a reasoning delimiter that arrived as its own token.
    private mutating func push(
        _ boundary: ReasoningEventEmitter.Boundary, emit: (TokenStreamEvent) -> Bool
    ) -> Bool {
        // A real implicit-end token (e.g. the protocol's tool-call token) means
        // the model switched to its call path: a rehearsal held from a spelled
        // opener is ordinary thinking text, not a call to classify. Doctrine
        // split (merged 0.33.0): upstream #563 pins spelled delimiters as text;
        // Phase 53 pins rehearsal-only streams as structured/rejected, which
        // still happens on response start and at finish.
        if case .implicitEnd = boundary, let held = releaseReasoningToolAttempt() {
            guard emit(.reasoning(held)) else { return false }
        }
        // Text still held by the detokenizer and the stop filter precedes the boundary.
        if let text = detokenizer.finish(), !process(text, emit: emit) { return false }
        detokenizer = tokenizer.makeStreamingDetokenizer()
        if let text = stopStringFilter.finish(), !route(text, emit: emit) { return false }

        // The delimiter is never decoded, so offer its text to the stop filter here:
        // a caller may stop on it. Whatever the filter passes or holds is dropped.
        guard let delimiter = reasoningTokens?.text(of: boundary) else { return true }
        if stopStringFilter.process(delimiter).stopped {
            _ = emit(.stop)
            return false
        }
        stopStringFilter.buffer = ""

        guard let segments = reasoningEmitter?.process(boundary: boundary) else { return true }
        return emitSegments(segments, emit: emit)
    }

    private mutating func process(_ chunk: String, emit: (TokenStreamEvent) -> Bool) -> Bool {
        let result = stopStringFilter.process(chunk)
        if let text = result.text, !route(text, emit: emit) {
            return false
        }
        if result.stopped {
            _ = emit(.stop)
            return false
        }
        return true
    }

    mutating func finish(emit: (TokenStreamEvent) -> Bool) -> Bool {
        if let text = detokenizer.finish(), !process(text, emit: emit) { return false }
        if let text = stopStringFilter.finish(), !route(text, emit: emit) {
            return false
        }

        // Drain the reasoning scanner before the tool processor's EOS: an end
        // delimiter that never arrives as text must not strand held-back thinking,
        // and what it flushes may still be response text the parser has to see.
        if let segments = reasoningEmitter?.finalize(),
            !emitSegments(segments, emit: emit)
        {
            return false
        }

        // Classify a tool attempt still held from the reasoning span before EOS:
        // it must surface as a structured call or an explicit rejection, never
        // silently disappear with the stream.
        if let attempt = finalizeReasoningToolAttempt() {
            guard emit(attempt) else { return false }
        }

        return emitOutputs(toolCallProcessor.processEOSOutputs(), emit: emit)
    }

    /// Splits reasoning off first and lets only non-reasoning text reach the
    /// tool-call parser: a model that writes `<|tool_call>` inside its scratchpad
    /// would otherwise produce a phantom tool call.
    private mutating func route(_ text: String, emit: (TokenStreamEvent) -> Bool) -> Bool {
        guard let segments = reasoningEmitter?.process(text) else {
            return emitOutputs(toolCallProcessor.processChunkOutputs(text), emit: emit)
        }
        return emitSegments(segments, emit: emit)
    }

    /// `mutating` so a later change that advances the emitter here cannot write to a
    /// copy; today it touches only the class-typed processor.
    private mutating func emitSegments(
        _ segments: [ReasoningEventEmitter.Segment],
        emit: (TokenStreamEvent) -> Bool
    ) -> Bool {
        for segment in segments {
            switch segment {
            case .reasoning(let reasoning):
                guard handleReasoningSegment(reasoning, emit: emit) else { return false }
            case .response(let response):
                let outputs = toolCallProcessor.processChunkOutputs(response)
                if outputs.contains(where: { if case .toolCall = $0 { return true }; return false }),
                    let held = releaseReasoningToolAttempt()
                {
                    // A real parsed call supersedes the held rehearsal: it was
                    // thinking text. Only classify when no call ever arrives.
                    guard emit(.reasoning(held)) else { return false }
                } else if let attempt = finalizeReasoningToolAttempt() {
                    guard emit(attempt) else { return false }
                }
                guard emitOutputs(outputs, emit: emit) else { return false }
            }
        }
        return true
    }

    private static let explicitToolOpeners = [
        "<tool_call>", "<|tool_call>", "<function=", "[TOOL_CALLS]",
    ]

    /// Routes a reasoning segment: prose before an explicit tool-protocol
    /// opener streams as reasoning; from the opener onward the text is held
    /// and ultimately classified, so a rehearsed call never leaks as text.
    private mutating func handleReasoningSegment(
        _ text: String, emit: (TokenStreamEvent) -> Bool
    ) -> Bool {
        if reasoningToolAttempt != nil {
            reasoningToolAttempt?.append(text)
            return true
        }
        var best: (Range<String.Index>, String)?
        for opener in Self.explicitToolOpeners {
            if let range = text.range(of: opener),
                best == nil || range.lowerBound < best!.0.lowerBound
            {
                best = (range, opener)
            }
        }
        guard let best else {
            return emit(.reasoning(text))
        }
        let head = String(text[..<best.0.lowerBound])
        reasoningToolAttempt = String(text[best.0.lowerBound...])
        if !head.isEmpty {
            return emit(.reasoning(head))
        }
        return true
    }

    /// Takes the held attempt without classifying it. Used when a real tool-call
    /// signal (the implicit-end token or a parsed call) supersedes the
    /// rehearsal: it streams as ordinary reasoning.
    private mutating func releaseReasoningToolAttempt() -> String? {
        defer { reasoningToolAttempt = nil }
        return reasoningToolAttempt
    }

    /// Classifies the held attempt. Doctrine: reasoning-span calls surface as a
    /// structured call when parseable (client-decides) and otherwise as an
    /// explicit rejection — never silently dropped, never leaked as prose.
    private mutating func finalizeReasoningToolAttempt() -> TokenStreamEvent? {
        guard let attempt = reasoningToolAttempt else { return nil }
        reasoningToolAttempt = nil
        // Client-decides: a parseable attempt is delivered as a structured
        // call through a dialect-matched processor, malformed closers and all.
        let dialect: ToolCallFormat
        if attempt.contains("<function=") {
            dialect = .qwen35
        } else if attempt.contains("[TOOL_CALLS]") {
            dialect = .mistral
        } else if attempt.contains("<|tool_call>") {
            dialect = .gptOSS
        } else {
            dialect = .json
        }
        // Qwen-family checkpoints intermittently emit the plural closer
        // `</tool_calls>` for a `<tool_call>` frame; normalize the known
        // hallucinations so the lenient parse can complete.
        var normalizedAttempt = attempt
            .replacingOccurrences(of: "</tool_calls>", with: "</tool_call>")
        normalizedAttempt += dialect.closingStopStrings.joined()
        var promotionProcessor = ToolCallProcessor(
            format: dialect, tools: tools, toolCallPolicy: toolCallPolicy)
        var outputs = promotionProcessor.processChunkOutputs(normalizedAttempt)
        if !outputs.contains(where: { if case .toolCall = $0 { return true }; return false }) {
            for frame in dialect.closingStopStrings {
                outputs += promotionProcessor.processChunkOutputs(frame)
            }
            outputs += promotionProcessor.processEOSOutputs()
        }
        if let call = outputs.compactMap({ output -> ToolCall? in
            if case let .toolCall(call) = output { return call }
            return nil
        }).first {
            return .toolCall(call)
        }

        // Unparseable: explicit rejection with bounded preview.
        let stripped = Self.explicitToolOpeners.reduce(attempt) { partial, opener in
            partial.replacingOccurrences(of: opener, with: "")
        }
        let nameCandidate =
            stripped
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .prefix { $0.isLetter || $0.isNumber || $0 == "_" || $0 == "-" }
        return .rejectedToolCall(
            RejectedToolCall(
                reason: nameCandidate.isEmpty ? .missingToolName : .undeclaredTool,
                format: format,
                toolName: nameCandidate.isEmpty ? nil : String(nameCandidate),
                rawText: attempt
            ))
    }

    /// Maps ordered processor outputs onto stream events, keeping response text,
    /// accepted calls, and rejected calls in the order the model emitted them.
    private func emitOutputs(
        _ outputs: [ToolCallProcessor.Output],
        emit: (TokenStreamEvent) -> Bool
    ) -> Bool {
        for output in outputs {
            let event: TokenStreamEvent =
                switch output {
                case .response(let text): .response(text)
                case .toolCall(let call): .toolCall(call)
                case .rejectedToolCall(let rejection): .rejectedToolCall(rejection)
                }
            guard emit(event) else { return false }
        }
        return true
    }
}
