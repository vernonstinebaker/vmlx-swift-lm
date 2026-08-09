import MLX
import MLXLLM
import Testing
import Foundation

@testable import MLXLMCommon

@Suite("Compiled batch decode", .serialized)
struct CompiledBatchDecodeTests {
    @Test(
        "Compiled decode advances fixed-shape caches",
        .enabled(if: ProcessInfo.processInfo.environment["MLX_RUN_COMPILED_TESTS"] == "1")
    )
    func compiledDecodeAdvancesCache() {
        let configuration = LlamaConfiguration(
            hiddenSize: 64,
            hiddenLayers: 2,
            intermediateSize: 128,
            attentionHeads: 8,
            rmsNormEps: 1e-5,
            vocabularySize: 128,
            kvHeads: 4)
        let model = LlamaModel(configuration)
        let prompt = MLXArray(Int32(1) ..< Int32(9))
        let cache = model.newCache(parameters: nil)
        let prefill = model(
            LMInput.Text(tokens: prompt)[text: .newAxis],
            cache: cache,
            state: nil)
        MLX.eval(prefill.logits)
        MLX.eval(cache)

        let compiledCache = cache.map { CompilableKVCache(from: $0, maxLength: 64) }
        MLX.eval(compiledCache)
        let nextToken = argMax(prefill.logits[0 ..< 1, -1, 0...], axis: -1)
        let decode = BatchCompile.compileForward(model: model, cache: compiledCache)
        let logits = decode([nextToken])[0]
        MLX.eval(logits)

        #expect(logits.shape == [1, 1, 128])
        #expect(compiledCache.allSatisfy { $0.offset == 9 })
    }
}
