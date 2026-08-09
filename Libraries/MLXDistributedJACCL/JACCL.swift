import Foundation
import MLX

/// Thin Swift binding around MLX's JACCL distributed backend (RDMA over
/// Thunderbolt 5 / TB4 on Apple Silicon, macOS 26.3+). Phase 4 ships a
/// single capability probe (`isAvailable`) so callers can detect whether
/// the upstream C++ + librdma + verbs SDK headers are wired correctly on
/// the host. Real `Group` lifecycle + collectives land in Phase 5.
public enum JACCL {
    /// Returns true when the JACCL backend can initialise on this host
    /// — i.e. `librdma.dylib` loads, the verbs ABI matches, and at least
    /// one IBV port is queryable. Note: returning `true` does NOT imply
    /// a peer is currently reachable; only that the local stack is sound.
    public static func isAvailable() -> Bool {
        return "jaccl".withCString { ptr in
            _vmlx_distributed_is_available(ptr)
        }
    }

    /// Same as `isAvailable()` but probes the global "any backend"
    /// gate, which returns true if any of jaccl/ring/mpi/nccl is
    /// currently available. Useful for detecting whether we're in a
    /// fully-distributed-capable build at all.
    public static func anyBackendAvailable() -> Bool {
        return _vmlx_distributed_is_available(nil)
    }
}

// MARK: - C symbol forward declarations

@_silgen_name("vmlx_distributed_is_available")
private func _vmlx_distributed_is_available(_ backend: UnsafePointer<CChar>?) -> Bool
