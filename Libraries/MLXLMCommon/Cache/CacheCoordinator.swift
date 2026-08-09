import Foundation

public enum KVQuantizationMode: Sendable, Equatable {
    case none
    case turboQuant(keyBits: Int, valueBits: Int)
}

public struct CacheCoordinatorConfig: Sendable {
    public var usePagedCache: Bool
    public var enableDiskCache: Bool
    public var pagedBlockSize: Int
    public var maxCacheBlocks: Int
    public var diskCacheMaxGB: Float
    public var diskCacheDir: URL?
    public var ssmMaxEntries: Int
    public var modelKey: String?
    public var defaultKVMode: KVQuantizationMode
    public var defaultMaxKVSize: Int?
    public var longPromptMultiplier: Double

    public init(
        usePagedCache: Bool = true,
        enableDiskCache: Bool = false,
        pagedBlockSize: Int = 64,
        maxCacheBlocks: Int = 1_000,
        diskCacheMaxGB: Float = 10,
        diskCacheDir: URL? = nil,
        ssmMaxEntries: Int = 50,
        modelKey: String? = nil,
        defaultKVMode: KVQuantizationMode = .none,
        defaultMaxKVSize: Int? = nil,
        longPromptMultiplier: Double = 2
    ) {
        self.usePagedCache = usePagedCache
        self.enableDiskCache = enableDiskCache
        self.pagedBlockSize = pagedBlockSize
        self.maxCacheBlocks = maxCacheBlocks
        self.diskCacheMaxGB = diskCacheMaxGB
        self.diskCacheDir = diskCacheDir
        self.ssmMaxEntries = ssmMaxEntries
        self.modelKey = modelKey
        self.defaultKVMode = defaultKVMode
        self.defaultMaxKVSize = defaultMaxKVSize
        self.longPromptMultiplier = longPromptMultiplier
    }

    public func resolveKVPolicy(
        kvMode: KVQuantizationMode,
        maxKVSize: Int?,
        promptTokenCount: Int
    ) -> (kvMode: KVQuantizationMode, maxKVSize: Int?) {
        let mode = kvMode == .none ? defaultKVMode : kvMode
        let size: Int?
        if let defaultMaxKVSize, maxKVSize == nil,
           Double(promptTokenCount) > Double(defaultMaxKVSize) * longPromptMultiplier
        {
            size = defaultMaxKVSize
        } else {
            size = maxKVSize
        }
        return (mode, size)
    }
}

public struct PagedCacheStats: Sendable {
    public let totalBlocks: Int
    public let allocatedBlocks: Int
    public let freeBlocks: Int
    public let cacheHits: Int
    public let cacheMisses: Int
}

public final class PagedCacheManager: @unchecked Sendable {
    public let blockSize: Int
    public let maxBlocks: Int
    public let modelKey: String?

    public init(blockSize: Int, maxBlocks: Int, modelKey: String? = nil) {
        self.blockSize = blockSize
        self.maxBlocks = maxBlocks
        self.modelKey = modelKey
    }

    public var stats: PagedCacheStats {
        .init(totalBlocks: maxBlocks, allocatedBlocks: 0, freeBlocks: max(0, maxBlocks - 1), cacheHits: 0, cacheMisses: 0)
    }

    public func clear() {}
}

public struct DiskCacheStats: Sendable {
    public let hits: Int
    public let misses: Int
}

public final class DiskCache: @unchecked Sendable {
    public var hits: Int { 0 }
    public var misses: Int { 0 }
    public func clear() {}
}

public final class SSMStateCache: @unchecked Sendable {
    public var reDerives: Int { 0 }
    public func clear() {}
}

public final class CacheCoordinator: @unchecked Sendable {
    public let config: CacheCoordinatorConfig
    public let pagedCache: PagedCacheManager?
    public let diskCache: DiskCache?
    public let ssmStateCache = SSMStateCache()

    public init(config: CacheCoordinatorConfig = .init()) {
        self.config = config
        pagedCache = config.usePagedCache
            ? PagedCacheManager(
                blockSize: config.pagedBlockSize,
                maxBlocks: config.maxCacheBlocks,
                modelKey: config.modelKey)
            : nil
        diskCache = config.enableDiskCache ? DiskCache() : nil
    }

    public func clear() {
        pagedCache?.clear()
        diskCache?.clear()
        ssmStateCache.clear()
    }
}
