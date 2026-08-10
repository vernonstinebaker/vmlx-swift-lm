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
        case .none, .autoregressive:
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
