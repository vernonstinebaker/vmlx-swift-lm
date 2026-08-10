import Foundation
import MLX
import MLXLLM
@testable import MLXLMCommon
import Testing

@Suite("Speculative strategy dispatch", .serialized)
struct SpecDecStrategyTests {
    @Test("Full-prefill strategies preserve explicit cache configuration")
    func cachePolicyUsesCanonicalGenerationWhenNeeded() throws {
        #expect(try SpecDecStrategyCachePolicy.usesFullReprefill(
            cache: nil,
            parameters: .init(draftStrategy: .dflash(
                drafterPath: URL(filePath: "/tmp/dflash"), blockSize: 2
            ))
        ))
        #expect(try !(SpecDecStrategyCachePolicy.usesFullReprefill(
            cache: nil,
            parameters: .init(
                maxKVSize: 32,
                draftStrategy: .dflash(drafterPath: URL(filePath: "/tmp/dflash"), blockSize: 2)
            )
        )))
        #expect(try !(SpecDecStrategyCachePolicy.usesFullReprefill(
            cache: [KVCacheSimple()],
            parameters: .init(draftStrategy: .dflash(
                drafterPath: URL(filePath: "/tmp/dflash"), blockSize: 2
            ))
        )))
    }

    @Test(
        "Selected strategies execute their real runtimes",
        .enabled(if: ProcessInfo.processInfo.environment["MLX_RUN_SPECDEC_TESTS"] == "1")
    )
    func selectedStrategiesExecuteTheirRuntimes() throws {
        let target = try makeTarget()
        let drafter = try makeDrafter()
        let resolver = SpecDecDrafterResolver(load: { _ in drafter })
        for strategy in [
            DraftStrategy.dflash(drafterPath: URL(filePath: "/tmp/dflash"), blockSize: 2),
            .ddtree(drafterPath: URL(filePath: "/tmp/dflash"), branchingBudget: 2, blockSize: 2)
        ] {
            var iterator = try SpecDecStrategyTokenIterator(
                input: .init(tokens: MLXArray([Int32(1), 2, 3])),
                target: target,
                parameters: .init(maxTokens: 2, temperature: 0, draftStrategy: strategy),
                resolver: resolver
            )

            #expect(iterator.runtimeKind == strategy.kindName)
            #expect(iterator.next() != nil)
            #expect(iterator.tokenCount == 1)
        }
    }

    private func makeTarget() throws -> Qwen3Model {
        let json = """
        {
          "hidden_size": 16,
          "num_hidden_layers": 2,
          "intermediate_size": 32,
          "num_attention_heads": 2,
          "num_key_value_heads": 2,
          "head_dim": 8,
          "rms_norm_eps": 0.000001,
          "vocab_size": 32
        }
        """
        return try Qwen3Model(JSONDecoder().decode(Qwen3Configuration.self, from: Data(json.utf8)))
    }

    private func makeDrafter() throws -> DFlashDraftModel {
        let json = """
        {
          "hidden_size": 16,
          "num_hidden_layers": 1,
          "num_attention_heads": 2,
          "num_key_value_heads": 2,
          "head_dim": 8,
          "intermediate_size": 32,
          "block_size": 2,
          "dflash_config": {
            "mask_token_id": 0,
            "target_layer_ids": [0],
            "block_size": 2
          }
        }
        """
        return try DFlashDraftModel(
            JSONDecoder().decode(DFlashDrafterConfiguration.self, from: Data(json.utf8))
        )
    }
}
