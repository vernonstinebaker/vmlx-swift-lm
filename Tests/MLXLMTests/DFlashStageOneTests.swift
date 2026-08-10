import Foundation
import MLX
import MLXLLM
@testable import MLXLMCommon
import Testing

@Suite("DFlash stage one", .serialized)
struct DFlashStageOneTests {
    @Test("DFlash checkpoint config supports nested block size and RoPE")
    func decodesNestedCheckpointConfiguration() throws {
        let data = Data("""
        {
          "hidden_size": 5120,
          "num_hidden_layers": 6,
          "num_attention_heads": 32,
          "num_key_value_heads": 8,
          "head_dim": 128,
          "intermediate_size": 17408,
          "rms_norm_eps": 0.000001,
          "rope_parameters": { "rope_theta": 10000000 },
          "dflash_config": {
            "block_size": 16,
            "mask_token_id": 248077,
            "target_layer_ids": [1, 10, 18, 27, 35, 44, 52, 61]
          }
        }
        """.utf8)

        let configuration = try JSONDecoder().decode(DFlashDrafterConfiguration.self, from: data)

        #expect(configuration.blockSize == 16)
        #expect(configuration.ropeTheta == 10_000_000)
        #expect(configuration.dflashConfig.targetLayerIDs == [1, 10, 18, 27, 35, 44, 52, 61])
    }

    @Test(
        "DFlash loader accepts the configured local checkpoint",
        .enabled(if: ProcessInfo.processInfo.environment["DFLASH_REAL_DRAFTER_PATH"] != nil)
    )
    func loadsLocalCheckpoint() throws {
        let path = try #require(ProcessInfo.processInfo.environment["DFLASH_REAL_DRAFTER_PATH"])
        let model = try DFlashDrafterLoader.load(from: URL(fileURLWithPath: path))

        #expect(model.config.blockSize == 16)
        #expect(model.config.dflashConfig.targetLayerIDs.count == 8)
    }

    @Test(
        "DFlash linear output matches greedy Qwen3",
        .enabled(if: ProcessInfo.processInfo.environment["MLX_RUN_SPECDEC_TESTS"] == "1")
    )
    func linearOutputMatchesGreedyQwen3() throws {
        let targetConfiguration = try JSONDecoder().decode(Qwen3Configuration.self, from: Data("""
        {
          "hidden_size": 128,
          "num_hidden_layers": 4,
          "intermediate_size": 256,
          "num_attention_heads": 4,
          "num_key_value_heads": 2,
          "head_dim": 32,
          "rms_norm_eps": 0.000001,
          "vocab_size": 512
        }
        """.utf8))
        let drafterConfiguration = try JSONDecoder().decode(DFlashDrafterConfiguration.self, from: Data("""
        {
          "hidden_size": 128,
          "num_hidden_layers": 2,
          "num_attention_heads": 4,
          "num_key_value_heads": 2,
          "head_dim": 32,
          "intermediate_size": 256,
          "block_size": 4,
          "dflash_config": {
            "mask_token_id": 500,
            "target_layer_ids": [0, 2]
          }
        }
        """.utf8))
        let target = Qwen3Model(targetConfiguration)
        let prompt: [Int32] = [7, 11, 13, 17]
        let result = SpecDecRuntimeLinear.run(.init(
            target: target,
            drafter: DFlashDraftModel(drafterConfiguration),
            targetBlockIDs: [0, 2],
            maskTokenID: 500,
            inputIds: MLXArray(prompt).reshaped(1, prompt.count),
            maxNewTokens: 8
        ))

        var expected = prompt
        for _ in 0 ..< 8 {
            let logits = target(MLXArray(expected).reshaped(1, expected.count), cache: nil)
            MLX.eval(logits)
            expected.append(argMax(logits[0, -1, 0...], axis: -1).asType(.int32).item(Int32.self))
        }

        #expect(result.tokenIds == expected)
        #expect(!result.acceptanceLengths.isEmpty)
    }
}
