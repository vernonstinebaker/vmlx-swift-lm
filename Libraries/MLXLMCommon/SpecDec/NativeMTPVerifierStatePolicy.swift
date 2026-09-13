// Ported from osaurus-ai/vmlx-swift@main Libraries/MLXLMCommon/MTPRuntime.swift
// (verifier-state policy section). Divergence: upstream hosts this enum inside
// MTPRuntime.swift, which this line does not carry; the policy is extracted
// verbatim so DFlash2 + native MTP share one verifier-mode source of truth.
import Foundation

public enum NativeMTPVerifierStatePolicy {
    public enum Mode: String, Sendable, Equatable {
        case captureCommit = "capture_commit"
        case strictCapture = "strict_capture"
        case lazyRepair = "lazy_repair"
        /// DFlash 2's rollback mode: recurrent layers stash their verify
        /// INPUTS (references, zero cost) instead of recording a state per
        /// prefix length (which re-runs the scan per prefix and flushes
        /// the graph per record). Rollback replays only the accepted rows,
        /// one kernel per layer, only when a rejection happens.
        case inputCapture = "input_capture"
        /// Compile-compatible variant of `inputCapture`: recurrent layers
        /// write their verify inputs into FIXED staging slots on the cache
        /// (in place, so `compile()` tracks them as state outputs) and do
        /// NOT touch their committed state or offset. A host-side commit
        /// applies the accepted prefix after acceptance — replaying n rows
        /// from the untouched pre-verify state, or copying the staged final
        /// state on a full accept. `inputCapture`'s host-struct stash cannot
        /// survive a compiled trace: the assignment runs once at trace time
        /// and would pin trace tracers forever.
        case inputCaptureStaged = "input_capture_staged"
    }

    @TaskLocal public static var requestVerifierMode: String?

    public static var mode: Mode {
        mode(for: requestVerifierMode)
    }

    public static func mode(for requestedMode: String?) -> Mode {
        let env = ProcessInfo.processInfo.environment
        let raw =
            (requestedMode
                ?? env["VMLX_NATIVE_MTP_STATE_COMMIT"]
                ?? env["VMLINUX_NATIVE_MTP_STATE_COMMIT"]
                ?? env["VMLX_NATIVE_MTP_HYBRID_VERIFY"]
                ?? env["VMLINUX_NATIVE_MTP_HYBRID_VERIFY"]
                ?? "")
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased()
            .replacingOccurrences(of: "-", with: "_")
        if raw.isEmpty {
            return .lazyRepair
        }
        switch raw {
        case "chunk_repair", "chunk_step_repair":
            return .captureCommit
        case "chunk_lazy_repair", "lazy_repair", "lazy", "fast_lazy":
            return .lazyRepair
        case "input_capture", "verify_input_capture":
            return .inputCapture
        case "input_capture_staged", "verify_input_capture_staged":
            return .inputCaptureStaged
        case "chunk_fast", "fast", "capture_commit", "chunk_commit":
            return .captureCommit
        case "sequential", "sequential_repair", "strict", "strict_capture":
            return .strictCapture
        default:
            return .strictCapture
        }
    }

    public static func withVerifierMode<R>(
        _ verifierMode: String?,
        operation: () throws -> R
    ) rethrows -> R {
        try $requestVerifierMode.withValue(verifierMode, operation: operation)
    }

    public static var shouldRecordAcceptedPrefixStates: Bool {
        mode != .lazyRepair && mode != .inputCapture && mode != .inputCaptureStaged
    }

    /// Whether recurrent layers should stash their verify inputs for the
    /// one-shot lazy rollback (DFlash 2).
    public static var shouldStashVerifyInputs: Bool {
        mode == .inputCapture
    }

    /// Whether recurrent layers should write verify inputs + final state
    /// into the cache's fixed staging slots and leave committed state
    /// untouched (compiled DFlash 2 verify).
    public static var shouldStageVerifyInputs: Bool {
        mode == .inputCaptureStaged
    }

    public static var shouldRoundGDNStateEachVerifierStep: Bool {
        // strictCapture only. inputCapture deliberately does NOT round:
        // recurrent state is float32 end-to-end, so a T=n scan is
        // bit-identical to n chained T=1 scans (audited at 0.0 maxdiff on
        // JANG_6D), and per-step rounding would CHANGE verify numerics
        // relative to the plain decode path instead of matching it.
        mode == .strictCapture
    }
}
