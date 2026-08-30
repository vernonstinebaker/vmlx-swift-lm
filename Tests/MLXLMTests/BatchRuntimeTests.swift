import Testing
import MLX

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

    @Test("Continuation state stacks and splits on the batch axis")
    func continuationStateStacksAlongBatch() {
        let key = LMOutput.Key<MLXArray>("rope_deltas")
        var first = LMOutput.State()
        var second = LMOutput.State()
        first[key] = MLXArray([Int32(4)])
        second[key] = MLXArray([Int32(7)])

        let stacked = try #require(LMOutput.State.stackedAlongBatch([first, second]))
        #expect(stacked[key]?.shape == [2])
        #expect(stacked[key]?.asArray(Int32.self) == [4, 7])

        let split = try #require(stacked.splitAlongBatch(count: 2))
        #expect(split[0][key]?.asArray(Int32.self) == [4])
        #expect(split[1][key]?.asArray(Int32.self) == [7])
    }

    @Test("Non-batch-leading state keys are omitted so stacking can still succeed")
    func continuationStateOmitsNonBatchLeadingKeys() {
        let rope = LMOutput.Key<MLXArray>("rope_deltas")
        let positions = LMOutput.Key<MLXArray>("precomputed_position_ids")
        var first = LMOutput.State()
        var second = LMOutput.State()
        first[rope] = MLXArray([Int32(1)])
        second[rope] = MLXArray([Int32(2)])
        first[positions] = MLXArray(0 ..< 6).reshaped(3, 1, 2)
        second[positions] = MLXArray(0 ..< 6).reshaped(3, 1, 2)

        let stacked = try #require(LMOutput.State.stackedAlongBatch([first, second]))
        #expect(stacked[rope]?.asArray(Int32.self) == [1, 2])
        #expect(stacked[positions] == nil)
    }
}
