import Foundation
import MLX  // pulls Cmlx symbols into the link

/// Swift wrapper around an MLX `mlx_distributed_group` handle. Created
/// once per process via `Group.init(strict:backend:)`; subsequent
/// `split` returns child groups that share the same lifecycle.
///
/// Phase 5 scope:
/// - rank, size, split queryable.
/// - Lifecycle: leaks on shutdown (mlx-c does NOT expose a public
///   `_free` for groups; the C++ Group is shared_ptr-managed but the
///   wrapper allocates with `new` and the only deleter is private).
///   Single Group per process is the expected pattern, so this
///   shouldn't matter in practice.
public struct Group: Sendable {
    /// Opaque mlx_distributed_group handle (just a void* ctx wrapper).
    public let handle: MLXDistributedGroupHandle

    /// Initialise the global distributed group for this process.
    /// - Parameters:
    ///   - strict: when true, throws if the requested backend can't init
    ///     (env vars missing, etc). When false, returns a trivial size-1
    ///     group on failure — same semantics as Python's
    ///     `mx.distributed.init()`.
    ///   - backend: optional backend hint ("jaccl", "ring", "mpi",
    ///     "nccl"). nil lets MLX pick.
    public init(strict: Bool = false, backend: String? = nil) {
        if let backend {
            self.handle = backend.withCString { bk in
                MLXDistributedGroupHandle(_vmlx_group_init(strict, bk))
            }
        } else {
            self.handle = MLXDistributedGroupHandle(_vmlx_group_init(strict, nil))
        }
    }

    /// Number of ranks in this group.
    public var size: Int {
        Int(_vmlx_group_size(handle.raw))
    }

    /// Local rank within this group.
    public var rank: Int {
        Int(_vmlx_group_rank(handle.raw))
    }

    /// Returns true if this group has more than one rank — i.e. real
    /// multi-host work is happening. False means a no-op "group of 1".
    public var isMultiRank: Bool { size > 1 }

    /// Split into a sub-group; ranks with the same `color` end up in
    /// the same returned group, with `key` controlling rank order.
    /// On a size-1 group this is a no-op (mlx-c rejects splits of the
    /// trivial group; we return self rather than expose the empty
    /// handle that would result).
    public func split(color: Int, key: Int) -> Group {
        guard isMultiRank else { return self }
        let raw = _vmlx_group_split(handle.raw, Int32(color), Int32(key))
        return Group(handle: MLXDistributedGroupHandle(raw))
    }

    private init(handle: MLXDistributedGroupHandle) {
        self.handle = handle
    }
}

/// Sendable wrapper around the mlx_distributed_group `ctx` pointer.
public struct MLXDistributedGroupHandle: @unchecked Sendable {
    let raw: UnsafeMutableRawPointer?

    init(_ raw: UnsafeMutableRawPointer?) { self.raw = raw }
}

// MARK: - C symbol forward declarations
//
// mlx-swift doesn't export Cmlx as a public product, so these declarations
// bind to the vMLX C shim target.

@_silgen_name("vmlx_group_init")
private func _vmlx_group_init(
    _ strict: Bool, _ backend: UnsafePointer<CChar>?
) -> UnsafeMutableRawPointer?

@_silgen_name("vmlx_group_rank")
private func _vmlx_group_rank(_ g: UnsafeMutableRawPointer?) -> Int32

@_silgen_name("vmlx_group_size")
private func _vmlx_group_size(_ g: UnsafeMutableRawPointer?) -> Int32

@_silgen_name("vmlx_group_split")
private func _vmlx_group_split(
    _ g: UnsafeMutableRawPointer?, _ color: Int32, _ key: Int32
) -> UnsafeMutableRawPointer?
