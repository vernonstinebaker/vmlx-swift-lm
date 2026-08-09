import Testing

@testable import MLXLMCommon

@Suite("Batch runtime")
struct BatchRuntimeTests {
    @Test("Cache configuration applies only absent request values")
    func cacheConfigurationResolvesDefaults() {
        let configuration = CacheCoordinatorConfig(
            defaultKVMode: .turboQuant(keyBits: 3, valueBits: 2),
            defaultMaxKVSize: 4_096,
            longPromptMultiplier: 1.5)

        let inherited = configuration.resolveKVPolicy(
            kvMode: .none, maxKVSize: nil, promptTokenCount: 7_000)
        #expect(inherited.kvMode == .turboQuant(keyBits: 3, valueBits: 2))
        #expect(inherited.maxKVSize == 4_096)

        let explicit = configuration.resolveKVPolicy(
            kvMode: .turboQuant(keyBits: 8, valueBits: 4),
            maxKVSize: 2_048,
            promptTokenCount: 7_000)
        #expect(explicit.kvMode == .turboQuant(keyBits: 8, valueBits: 4))
        #expect(explicit.maxKVSize == 2_048)
    }

    @Test("Batch request IDs are unique and concise")
    func batchRequestIDsAreUnique() {
        let first = BatchRequestID()
        let second = BatchRequestID()

        #expect(first != second)
        #expect(first.description.count == 8)
        #expect(second.description.count == 8)
    }
}
