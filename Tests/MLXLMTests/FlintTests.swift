import Foundation
import MLX
import MLXLMCommon
import Testing

@testable import MLXLLM

struct FlintTests {
    private let configJSON = """
        {
          "model_type": "flint",
          "hidden_size": 16,
          "num_hidden_layers": 2,
          "intermediate_size": 32,
          "num_attention_heads": 2,
          "num_key_value_heads": 1,
          "head_dim": 8,
          "num_experts": 2,
          "num_experts_per_tok": 1,
          "moe_intermediate_size": 8,
          "shared_expert_intermediate_size": 8,
          "mlp_only_layers": [0],
          "layer_types": ["full_attention", "sliding_attention"],
          "sliding_window": 4,
          "rms_norm_eps": 1e-6,
          "vocab_size": 32,
          "rope_parameters": {"rope_theta": 10000.0, "rope_type": "default"},
          "max_position_embeddings": 128,
          "tie_word_embeddings": false
        }
        """

    @Test("decodes the published Flint config shape")
    func decodesConfiguration() throws {
        let config = try JSONDecoder().decode(
            FlintConfiguration.self, from: Data(configJSON.utf8))

        #expect(config.modelType == "flint")
        #expect(config.hiddenLayers == 2)
        #expect(config.layerTypes == ["full_attention", "sliding_attention"])
        #expect(config.numExperts == 2)
        #expect(config.ropeTheta == 10000)
        #expect(config.maxPositionEmbeddings == 128)
    }

    @Test("builds full and sliding caches by layer type")
    func cacheTopology() throws {
        let config = try JSONDecoder().decode(
            FlintConfiguration.self, from: Data(configJSON.utf8))
        let model = FlintModel(config)
        let caches = try model.newCache(parameters: nil)

        #expect(caches.count == 2)
        #expect(caches[0] is KVCacheSimple)
        #expect(caches[1] is RotatingKVCache)
    }

    @Test("views packed uint8 expert weights as uint32")
    func sanitizePackedExperts() throws {
        let config = try JSONDecoder().decode(
            FlintConfiguration.self, from: Data(configJSON.utf8))
        let model = FlintModel(config)
        let packed = MLXArray.zeros([4], dtype: .uint8)
        let scales = MLXArray.zeros([4], dtype: .bfloat16)
        let untouched = MLXArray.zeros([4], dtype: .uint8)

        let result = model.sanitize(weights: [
            "model.layers.1.mlp.switch_mlp.gate_proj.weight": packed,
            "model.layers.1.mlp.switch_mlp.gate_proj.scales": scales,
            "model.layers.1.router.weight": untouched,
        ])

        #expect(result["model.layers.1.mlp.switch_mlp.gate_proj.weight"]?.dtype == .uint32)
        #expect(result["model.layers.1.router.weight"]?.dtype == .uint8)
    }
}
