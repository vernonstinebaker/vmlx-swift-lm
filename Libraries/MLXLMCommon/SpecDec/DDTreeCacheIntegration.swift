public enum DDTreeCacheIntegrationError: Error, Equatable {
    case untrimmableCache
    case inconsistentTrim(expected: Int, actual: Int)
}

public enum DDTreeCacheIntegration {
    public static func fork(_ cache: [KVCache]) -> [KVCache] {
        cache.map { $0.copy() }
    }

    @discardableResult
    public static func retainAcceptedPath(
        in cache: [KVCache],
        compilation: DDTreeCompilation,
        verification: DDTreeVerification
    ) throws -> Int {
        let discarded = try discardedNodeCount(
            compilation: compilation,
            verification: verification
        )
        guard discarded > 0 else { return 0 }
        guard canTrimPromptCache(cache) else {
            throw DDTreeCacheIntegrationError.untrimmableCache
        }
        let trimmed = trimPromptCache(cache, numTokens: discarded)
        guard trimmed == discarded else {
            throw DDTreeCacheIntegrationError.inconsistentTrim(expected: discarded, actual: trimmed)
        }
        return trimmed
    }

    @discardableResult
    package static func retainAcceptedPath(
        in cacheStorage: KVCacheStorage,
        compilation: DDTreeCompilation,
        verification: DDTreeVerification
    ) throws -> Int {
        let discarded = try discardedNodeCount(
            compilation: compilation,
            verification: verification
        )
        guard discarded > 0 else { return 0 }
        guard canTrimPromptCache(cacheStorage.cache) else {
            throw DDTreeCacheIntegrationError.untrimmableCache
        }
        let trimmed = cacheStorage.trim(discarded)
        guard trimmed == discarded else {
            throw DDTreeCacheIntegrationError.inconsistentTrim(expected: discarded, actual: trimmed)
        }
        cacheStorage.plan.apply(to: cacheStorage)
        return trimmed
    }

    private static func discardedNodeCount(
        compilation: DDTreeCompilation,
        verification: DDTreeVerification
    ) throws -> Int {
        guard compilation.tree.nodes.count == verification.treeNodeCount else {
            throw DDTreeCacheIntegrationError.inconsistentTrim(
                expected: compilation.tree.nodes.count,
                actual: verification.treeNodeCount
            )
        }
        return verification.discardedNodeCount
    }
}
