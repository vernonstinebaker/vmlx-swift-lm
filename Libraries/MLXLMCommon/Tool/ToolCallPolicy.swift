// Copyright © 2026 Apple Inc.

/// Rules for recovering and validating generated tool calls.
/// Native parser requirements and declared-tool authorization always apply.
public struct ToolCallPolicy: Hashable, Sendable {
    /// Bounded recovery of calls emitted in a different dialect.
    /// Applies to text formats; framed token protocols use their native parser.
    public var recovery: ToolCallRecoveryPolicy

    /// Schema enforcement after unambiguous argument normalization.
    public var validation: ToolCallValidationPolicy

    /// What happens to a parsed call whose function name is not in the
    /// request's declared tools.
    ///
    /// Divergence note (LLMServerPlus): the default is `.surfaceToClient` —
    /// the model's intent arrives as a structured `tool_calls` payload and
    /// the client maps, executes, or displays it. Product decision
    /// 2026-09-17; `.rejectUndeclared` preserves upstream #548 behavior.
    /// Exit condition: upstream offers an equivalent knob, or the product
    /// reverts to count-only.
    public var authorization: ToolCallAuthorizationPolicy

    public init(
        recovery: ToolCallRecoveryPolicy = .conservative,
        validation: ToolCallValidationPolicy = .permissive,
        authorization: ToolCallAuthorizationPolicy = .surfaceToClient
    ) {
        self.recovery = recovery
        self.validation = validation
        self.authorization = authorization
    }
}

/// Client-decides vs reject: what an undeclared function name means.
public enum ToolCallAuthorizationPolicy: String, Hashable, Sendable, CaseIterable {
    /// Deliver the call as structured `tool_calls`; the client applies policy.
    case surfaceToClient
    /// Reject the call with reason `.undeclaredTool` (upstream #548 default).
    case rejectUndeclared
}

/// Controls schema enforcement independently of syntax recovery and tool-name
/// authorization. Both modes normalize unambiguous, schema-declared values.
public enum ToolCallValidationPolicy: String, Hashable, Sendable, CaseIterable {
    /// Reject proven schema violations. Unsupported schema assertions remain
    /// unknown. Enable this to check arguments before automatic tool dispatch.
    case strict
    /// Forward parsed arguments after normalization, even if they violate the
    /// schema. This is the default; applications validate arguments themselves.
    /// Native parser requirements and declared-tool authorization still apply.
    case permissive
}
