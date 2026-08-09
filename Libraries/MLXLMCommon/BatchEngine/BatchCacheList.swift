import MLX

public final class BatchCacheList: CacheList {
    private let slotCacheLists: [CacheList]
    private let batchedSubCaches: [KVCache]

    public init(slotCacheLists: [CacheList]) {
        precondition(!slotCacheLists.isEmpty)
        self.slotCacheLists = slotCacheLists
        let count = slotCacheLists[0].cacheCount
        precondition(slotCacheLists.allSatisfy { $0.cacheCount == count })
        batchedSubCaches = (0 ..< count).map { index in
            let caches = slotCacheLists.map { $0[index] }
            if let arrays = caches[0] as? ArraysCache {
                return BatchArraysCache(
                    slotCaches: [arrays] + caches.dropFirst().map { $0 as! ArraysCache })
            }
            return BatchKVCache(slotCaches: caches)
        }
        super.init(caches: batchedSubCaches)
        offset = slotCacheLists.map(\.offset).max() ?? 0
    }

    public func splitBack() {
        for cache in batchedSubCaches {
            (cache as? BatchArraysCache)?.splitBack()
        }
    }
}
