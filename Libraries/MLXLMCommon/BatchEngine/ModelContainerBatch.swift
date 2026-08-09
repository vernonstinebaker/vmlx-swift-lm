extension ModelContainer {
    public func makeBatchEngine(
        maxBatchSize: Int = 8,
        memoryPurgeInterval: Int = 256
    ) async -> BatchEngine {
        let coordinator = cacheCoordinator
        return await perform { context in
            nonisolated(unsafe) let context = context
            return BatchEngine(
                context: context,
                maxBatchSize: maxBatchSize,
                memoryPurgeInterval: memoryPurgeInterval,
                cacheCoordinator: coordinator)
        }
    }
}
