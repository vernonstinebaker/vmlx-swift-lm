import Foundation

public struct ResolvedDrafter: @unchecked Sendable {
    public let model: DFlashDraftModel
    public let targetBlockIDs: [Int]
    public let maskTokenID: Int32

    public init(model: DFlashDraftModel, targetBlockIDs: [Int], maskTokenID: Int32) {
        self.model = model
        self.targetBlockIDs = targetBlockIDs
        self.maskTokenID = maskTokenID
    }
}

public final class SpecDecDrafterResolver: @unchecked Sendable {
    public static let shared = SpecDecDrafterResolver()

    private let lock = NSLock()
    private var cache: [String: DFlashDraftModel] = [:]
    private let load: (URL) throws -> DFlashDraftModel

    public init(load: @escaping (URL) throws -> DFlashDraftModel = DFlashDrafterLoader.load) {
        self.load = load
    }

    public func resolve(strategy: DraftStrategy) throws -> ResolvedDrafter {
        let path: URL
        switch strategy {
        case let .dflash(drafterPath, _), let .ddtree(drafterPath, _, _):
            path = drafterPath.resolvingSymlinksInPath()
        case .dflash2, .none, .autoregressive:
            throw SpecDecStrategyError.unsupportedStrategy(strategy.kindName)
        }

        let model = try lock.withLock {
            if let model = cache[path.path] {
                return model
            }
            let model = try load(path)
            cache[path.path] = model
            return model
        }
        return .init(
            model: model,
            targetBlockIDs: model.config.dflashConfig.targetLayerIDs,
            maskTokenID: Int32(model.config.dflashConfig.maskTokenID)
        )
    }

    public func evictAll() {
        lock.withLock {
            cache.removeAll()
        }
    }
}

/// Resolved DFlash 2 drafter — a different model type than DFlash v1
/// (candidate-path selector + dynamic causal conv), cached separately.
public struct ResolvedDFlash2Drafter: @unchecked Sendable {
    public let model: DFlash2DraftModel
    public let targetLayerIDs: [Int]
    public let maskTokenID: Int32
    public let blockSize: Int
}

extension SpecDecDrafterResolver {
    /// Process-wide DFlash 2 drafter cache (a 27B drafter is ~3.8 GB).
    /// Delegates to DFlash2DrafterResolver.shared, which is lock-backed.
    public func resolveDFlash2(strategy: DraftStrategy) throws -> ResolvedDFlash2Drafter {
        guard case let .dflash2(drafterPath, _) = strategy else {
            throw SpecDecStrategyError.unsupportedStrategy(strategy.kindName)
        }
        let model = try Self.dflash2Resolver.drafter(at: drafterPath)
        return ResolvedDFlash2Drafter(
            model: model,
            targetLayerIDs: model.config.targetLayerIds,
            maskTokenID: Int32(model.config.maskTokenId),
            blockSize: model.config.blockSize)
    }

    private static let dflash2Resolver = DFlash2DrafterResolver.shared
}
