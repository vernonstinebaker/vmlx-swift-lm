import Foundation
import MLX
import MLXNN

public enum MLXExecutionCoordinator {
    private static let lock = NSLock()

    public static func withLock<T>(_ body: () throws -> T) rethrows -> T {
        try lock.withLock(body)
    }
}

public actor BatchEngine {
    public let maxBatchSize: Int
    public let memoryPurgeInterval: Int
    private let context: ModelContext
    private let cacheCoordinator: CacheCoordinator?
    private let stopTokenIDs: Set<Int>
    private var waitQueue: [BatchPendingRequest] = []
    private var activeSlots: [BatchSlot] = []
    private var loopTask: Task<Void, Never>?
    private var stepsSinceMemoryPurge = 0
    public private(set) var isShutdown = false

    public init(
        context: ModelContext,
        maxBatchSize: Int = 8,
        memoryPurgeInterval: Int = 256,
        cacheCoordinator: CacheCoordinator? = nil
    ) {
        precondition(maxBatchSize > 0)
        self.context = context
        self.maxBatchSize = maxBatchSize
        self.memoryPurgeInterval = memoryPurgeInterval
        self.cacheCoordinator = cacheCoordinator

        var stops = context.configuration.eosTokenIds
        if let eos = context.tokenizer.eosTokenId {
            stops.insert(eos)
        }
        if let unknown = context.tokenizer.unknownTokenId {
            stops.insert(unknown)
        }
        for token in context.configuration.extraEOSTokens {
            if let id = context.tokenizer.convertTokenToId(token) {
                stops.insert(id)
            }
        }
        stopTokenIDs = stops
    }

    @discardableResult
    public func submit(
        input: consuming sending LMInput,
        parameters: GenerateParameters
    ) -> (id: BatchRequestID, stream: AsyncStream<BatchGeneration>) {
        let (stream, continuation) = AsyncStream<BatchGeneration>.makeStream()
        let request = BatchPendingRequest(input: input, parameters: parameters, continuation: continuation)
        guard !isShutdown else {
            finish(request, reason: .cancelled)
            return (request.id, stream)
        }
        waitQueue.append(request)
        ensureLoopRunning()
        return (request.id, stream)
    }

    public func generate(
        input: consuming sending LMInput,
        parameters: GenerateParameters
    ) -> AsyncStream<Generation> {
        if parameters.draftStrategy?.usesBlockDiffusion == true {
            do {
                return try MLXLMCommon.generate(
                    input: input,
                    parameters: parameters,
                    context: context
                )
            } catch {
                let (stream, continuation) = AsyncStream<Generation>.makeStream()
                continuation.yield(.info(.init(
                    promptTokenCount: input.text.tokens.size,
                    generationTokenCount: 0,
                    promptTime: 0,
                    generationTime: 0,
                    stopReason: .cancelled
                )))
                continuation.finish()
                return stream
            }
        }
        let tokenizer = context.tokenizer
        let (id, tokenStream) = submit(input: input, parameters: parameters)
        let (stream, continuation) = AsyncStream<Generation>.makeStream()
        continuation.onTermination = { @Sendable _ in
            Task { await self.cancel(id) }
        }
        // Protocol-based models (e.g. Muse Glimmer's Onyx/ATEM, GPT-OSS Harmony)
        // frame reasoning/response/tool segments as control tokens. The naive
        // detokenizer would leak those control tokens into response text, so when
        // the model declares a framed protocol, route tokens through its decoder.
        let protocolDecoder = context.configuration.toolCallFormat?
            .makeProtocolTokenStreamDecoder(
                tokenizer: tokenizer,
                tools: nil,
                stopStrings: Set(parameters.extraStopStrings))
        Task {
            var detokenizer = NaiveStreamingDetokenizer(tokenizer: tokenizer)
            var stopFilter = StopStringFilter(stopStrings: Set(parameters.extraStopStrings))
            var streamDecoder = protocolDecoder
            var stopped = false
            for await event in tokenStream {
                switch event {
                case let .token(token):
                    if var decoder = streamDecoder {
                        let keepGoing = Self.routeProtocolToken(
                            token,
                            through: &decoder,
                            continuation: continuation,
                            stopFilter: &stopFilter,
                            stopped: &stopped)
                        streamDecoder = decoder
                        if !keepGoing, stopped {
                            self.cancel(id)
                        }
                    } else {
                        detokenizer.append(token: token)
                        if let text = detokenizer.next() {
                            let filtered = stopFilter.process(text)
                            if let text = filtered.text {
                                continuation.yield(.chunk(text))
                            }
                            if filtered.stopped {
                                stopped = true
                                self.cancel(id)
                            }
                        }
                    }
                case let .info(info):
                    if var decoder = streamDecoder {
                        Self.finishProtocol(
                            decoder: &decoder, stopped: stopped,
                            continuation: continuation)
                        streamDecoder = decoder
                    } else if !stopped, let tail = stopFilter.finish() {
                        continuation.yield(.chunk(tail))
                    }
                    detokenizer.startNewSegment()
                    continuation.yield(.info(stopped ? .init(
                        promptTokenCount: info.promptTokenCount,
                        generationTokenCount: info.generationTokenCount,
                        promptTime: info.promptTime,
                        generationTime: info.generateTime,
                        stopReason: .stop) : info))
                }
            }
            continuation.finish()
        }
        return stream
    }

    /// Routes a single token through a framed protocol decoder and emits the
    /// resulting `Generation` events. Returns `false` when the decoder signals
    /// stop. Mirrors the standard `TextToolTokenLoopHandler` event mapping.
    private nonisolated static func routeProtocolToken(
        _ token: Int,
        through decoder: inout any TokenStreamDecoder,
        continuation: AsyncStream<Generation>.Continuation,
        stopFilter: inout StopStringFilter,
        stopped: inout Bool
    ) -> Bool {
        var didStop = false
        let completed = decoder.push(token) { event in
            switch event {
            case .response(let text):
                let filtered = stopFilter.process(text)
                if let response = filtered.text {
                    _ = continuation.yield(.chunk(response))
                }
                if filtered.stopped {
                    didStop = true
                }
            case .reasoning(let text):
                _ = continuation.yield(.reasoning(text))
            case .toolCall(let toolCall):
                _ = continuation.yield(.toolCall(toolCall))
            case .protocolError, .stop:
                didStop = true
            }
            return !didStop
        }
        if didStop { stopped = true }
        return completed
    }

    private nonisolated static func finishProtocol(
        decoder: inout any TokenStreamDecoder,
        stopped: Bool,
        continuation: AsyncStream<Generation>.Continuation
    ) {
        _ = decoder.finish { event in
            switch event {
            case .response(let text):
                _ = continuation.yield(.chunk(text))
            case .reasoning(let text):
                _ = continuation.yield(.reasoning(text))
            case .toolCall(let toolCall):
                _ = continuation.yield(.toolCall(toolCall))
            case .protocolError, .stop:
                break
            }
            return !stopped
        }
    }

    public func cancel(_ id: BatchRequestID) {
        if let index = waitQueue.firstIndex(where: { $0.id == id }) {
            finish(waitQueue.remove(at: index), reason: .cancelled)
            return
        }
        if let index = activeSlots.firstIndex(where: { $0.id == id }) {
            finish(activeSlots[index], reason: .cancelled)
            activeSlots[index].isFinished = true
        }
    }

    public func shutdown() {
        guard !isShutdown else { return }
        isShutdown = true
        loopTask?.cancel()
        loopTask = nil
        for request in waitQueue {
            finish(request, reason: .cancelled)
        }
        waitQueue.removeAll()
        for slot in activeSlots {
            finish(slot, reason: .cancelled)
        }
        activeSlots.removeAll()
    }

    public var pendingCount: Int { waitQueue.count }
    public var activeCount: Int { activeSlots.count }
    public var isRunning: Bool { loopTask != nil }

    private func ensureLoopRunning() {
        guard loopTask == nil else { return }
        loopTask = Task { await schedulingLoop() }
    }

    private func schedulingLoop() async {
        while !Task.isCancelled, !isShutdown {
            guard !waitQueue.isEmpty || !activeSlots.isEmpty else { break }
            admitPendingRequests()
            step()
            activeSlots.removeAll { $0.isFinished }
            stepsSinceMemoryPurge += 1
            if stepsSinceMemoryPurge >= memoryPurgeInterval {
                MLXExecutionCoordinator.withLock { Memory.clearCache() }
                stepsSinceMemoryPurge = 0
            }
            await Task.yield()
        }
        loopTask = nil
    }

    private func admitPendingRequests() {
        while activeSlots.count < maxBatchSize, !waitQueue.isEmpty {
            let request = waitQueue.removeFirst()
            var parameters = request.parameters
            if parameters.maxKVSize == nil,
               let configuration = cacheCoordinator?.config,
               let defaultMaxKVSize = configuration.defaultMaxKVSize,
               Double(request.input.text.tokens.size) > Double(defaultMaxKVSize) * configuration.longPromptMultiplier
            {
                parameters.maxKVSize = defaultMaxKVSize
            }
            let cache = MLXExecutionCoordinator.withLock {
                context.model.newCache(parameters: parameters)
            }
            activeSlots.append(BatchSlot(request: request, cache: cache))
        }
    }

    private func step() {
        for index in activeSlots.indices where activeSlots[index].phase == .prefill {
            prefill(index)
        }
        let decodeIndices = activeSlots.indices.filter { activeSlots[$0].phase == .decode }
        if !decodeIndices.isEmpty {
            decode(decodeIndices)
        }
    }

    private func prefill(_ index: Int) {
        var slot = activeSlots[index]
        guard slot.maxTokens != 0 else {
            finish(slot, reason: .length)
            slot.isFinished = true
            activeSlots[index] = slot
            return
        }

        let firstToken: MLXArray
        do {
            firstToken = try MLXExecutionCoordinator.withLock {
                let prepared = try context.model.prepare(
                    slot.input, cache: slot.cache, state: nil, prefill: .init())
                let logits: MLXArray
                switch prepared {
                case let .tokens(text):
                    logits = context.model(text[text: .newAxis], cache: slot.cache, state: nil)
                        .logits[0 ..< 1, -1, 0...]
                case let .logits(output):
                    logits = output.logits[0 ..< 1, -1, 0...]
                }
                MLX.eval(slot.cache)
                MLX.eval(logits)
                return slot.sample(logits)
            }
        } catch {
            finish(slot, reason: .cancelled)
            slot.isFinished = true
            activeSlots[index] = slot
            return
        }

        if var processor = slot.processor {
            processor.prompt(slot.input.text.tokens)
            slot.processor = processor
        }
        slot.phase = .decode
        slot.decodeStartedAt = Date()
        promoteToCompiledDecode(&slot)
        emit(firstToken, to: &slot)
        activeSlots[index] = slot
    }

    private func decode(_ indices: [Int]) {
        if indices.count == 1, let forward = activeSlots[indices[0]].compiledForward {
            decodeCompiled(indices[0], forward: forward)
            return
        }

        let tokens = stacked(indices.compactMap { activeSlots[$0].nextToken }).reshaped(indices.count, 1)
        let layerCount = activeSlots[indices[0]].cache.count
        var arraysCaches: [BatchArraysCache] = []
        var cacheLists: [BatchCacheList] = []
        var caches: [KVCache] = []
        caches.reserveCapacity(layerCount)
        for layer in 0 ..< layerCount {
            let slotCaches = indices.map { activeSlots[$0].cache[layer] }
            if let cacheList = slotCaches[0] as? CacheList {
                let batched = BatchCacheList(
                    slotCacheLists: [cacheList] + slotCaches.dropFirst().map { $0 as! CacheList })
                cacheLists.append(batched)
                caches.append(batched)
                continue
            }
            if let arrays = slotCaches[0] as? ArraysCache {
                let batched = BatchArraysCache(
                    slotCaches: [arrays] + slotCaches.dropFirst().map { $0 as! ArraysCache })
                arraysCaches.append(batched)
                caches.append(batched)
                continue
            }
            caches.append(BatchKVCache(slotCaches: slotCaches))
        }
        nonisolated(unsafe) let forwardCaches = caches
        let logits = MLXExecutionCoordinator.withLock {
            let output = context.model(LMInput.Text(tokens: tokens), cache: forwardCaches, state: nil).logits
            MLX.eval(output)
            return output
        }
        arraysCaches.forEach { $0.splitBack() }
        cacheLists.forEach { $0.splitBack() }
        for (batchIndex, slotIndex) in indices.enumerated() {
            var slot = activeSlots[slotIndex]
            emit(slot.sample(logits[batchIndex ..< batchIndex + 1, 0, 0...]), to: &slot)
            activeSlots[slotIndex] = slot
        }
    }

    private func decodeCompiled(
        _ index: Int,
        forward: @Sendable ([MLXArray]) -> [MLXArray]
    ) {
        var slot = activeSlots[index]
        guard let nextToken = slot.nextToken else { return }
        let logits = MLXExecutionCoordinator.withLock {
            let result = forward([nextToken])
            precondition(result.count == 1)
            MLX.eval(result[0])
            return result[0][0 ..< 1, 0, 0...]
        }
        emit(slot.sample(logits), to: &slot)
        activeSlots[index] = slot
    }

    private func promoteToCompiledDecode(_ slot: inout BatchSlot) {
        guard maxBatchSize == 1,
              slot.parameters.enableCompiledBatchDecode,
              slot.parameters.compiledBatchBuckets.contains(1),
              slot.cache.allSatisfy({ $0 is KVCacheSimple })
        else {
            return
        }

        let requestedLength = slot.promptTokenCount + (slot.maxTokens ?? 0)
        let maxLength = max(256, requestedLength)
        let promoted = slot.cache.map { CompilableKVCache(from: $0, maxLength: maxLength) as KVCache }
        MLXExecutionCoordinator.withLock {
            MLX.eval(promoted)
        }
        slot.cache = promoted
        slot.compiledForward = BatchCompile.compileForward(
            model: context.model,
            cache: promoted.map { $0 as! CompilableKVCache })
    }

    private func emit(_ token: MLXArray, to slot: inout BatchSlot) {
        let tokenID = token.item(Int.self)
        if stopTokenIDs.contains(tokenID) {
            finish(slot, reason: .stop)
            slot.isFinished = true
            return
        }
        slot.continuation.yield(.token(tokenID))
        slot.generatedTokenCount += 1
        slot.nextToken = token
        if let maxTokens = slot.maxTokens, slot.generatedTokenCount >= maxTokens {
            finish(slot, reason: .length)
            slot.isFinished = true
        }
    }

    private func finish(_ request: BatchPendingRequest, reason: GenerateStopReason) {
        request.continuation.yield(.info(.init(
            promptTokenCount: request.input.text.tokens.size,
            generationTokenCount: 0,
            promptTime: 0,
            generationTime: 0,
            stopReason: reason)))
        request.continuation.finish()
    }

    private func finish(_ slot: BatchSlot, reason: GenerateStopReason) {
        let now = Date()
        slot.continuation.yield(.info(.init(
            promptTokenCount: slot.promptTokenCount,
            generationTokenCount: slot.generatedTokenCount,
            promptTime: (slot.decodeStartedAt ?? now).timeIntervalSince(slot.prefillStartedAt),
            generationTime: slot.decodeStartedAt.map { now.timeIntervalSince($0) } ?? 0,
            stopReason: reason)))
        slot.continuation.finish()
    }
}
