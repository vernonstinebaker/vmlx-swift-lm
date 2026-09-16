import CoreImage
import Foundation
import MLX
import MLXLMCommon
import Testing

@testable import MLXVLM

private struct Gemma4UnifiedTestTokenizer: Tokenizer {
    let vocabularySize: Int = 64
    let bosToken: String? = nil
    let eosToken: String? = nil
    let eosTokenId: Int? = 1
    let unknownToken: String? = nil
    let unknownTokenId: Int? = 0

    func encode(text: String, addSpecialTokens: Bool) -> [Int] {
        [31, 2]
    }

    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
        tokenIds.map(String.init).joined(separator: " ")
    }

    func convertTokenToId(_ token: String) -> Int? {
        Int(token)
    }

    func convertIdToToken(_ id: Int) -> String? {
        String(id)
    }

    func applyChatTemplate(
        messages: [[String: any Sendable]],
        tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] {
        [31, 2]
    }
}

@Suite("Gemma4 Unified multimodal restore")
struct Gemma4UnifiedTests {
    private func decodeConfig(_ json: String) throws -> Gemma4UnifiedConfiguration {
        try JSONDecoder.json5().decode(Gemma4UnifiedConfiguration.self, from: Data(json.utf8))
    }

    private func tinyTextJSON(vision: Bool, audio: Bool) -> String {
        """
        {
          "model_type": "gemma4_unified",
          "vocab_size": 32,
          "image_token_id": 31,
          "audio_token_id": 30,
          "video_token_id": 29,
          "text_config": {
            "hidden_size": 8,
            "num_hidden_layers": 1,
            "intermediate_size": 16,
            "num_attention_heads": 1,
            "num_key_value_heads": 1,
            "num_global_key_value_heads": 1,
            "head_dim": 8,
            "global_head_dim": 8,
            "vocab_size": 32,
            "vocab_size_per_layer_input": 0,
            "num_kv_shared_layers": 0,
            "hidden_size_per_layer_input": 0,
            "sliding_window": 8,
            "layer_types": ["full_attention"],
            "tie_word_embeddings": true
          },
          "vision_config": \(vision ? """
          {
            "model_patch_size": 4,
            "mm_embed_dim": 8,
            "mm_posemb_size": 8,
            "output_proj_dims": 8
          }
          """ : "null"),
          "audio_config": \(audio ? """
          {
            "output_proj_dims": 8,
            "hidden_size": 8
          }
          """ : "null")
        }
        """
    }

    @Test("Gemma4 Unified config decodes native vision, audio, and eoa defaults")
    func configDecoding() throws {
        let config = try decodeConfig(
            """
            {
              "model_type": "gemma4_unified",
              "eoa_token_index": 258883,
              "text_config": {},
              "vision_config": {},
              "audio_config": {}
            }
            """)

        #expect(config.modelType == "gemma4_unified")
        #expect(config.eoaTokenId == 258883)
        #expect(config.imageTokenId == 258_880)
        #expect(config.audioTokenId == 258_881)
        #expect(config.visionConfig?.modelPatchSize == 48)
        #expect(config.visionConfig?.mmEmbedDim == 3_840)
        #expect(config.audioConfig?.outputProjectionDimensions == 640)
    }

    @Test("Gemma4 Unified sanitize keeps vision keys and drops tied lm_head")
    func sanitizeKeepsVisionKeys() throws {
        let model = Gemma4Unified(try decodeConfig(tinyTextJSON(vision: true, audio: false)))
        let sanitized = model.sanitize(weights: [
            "lm_head.weight": MLXArray.zeros([8, 8]),
            "vision_embedder.patch_dense.weight": MLXArray.zeros([8, 48]),
            "embed_vision.embedding_projection.weight": MLXArray.zeros([8, 8]),
            "embed_audio.embedding_projection.weight": MLXArray.zeros([8, 8]),
        ])

        #expect(sanitized["lm_head.weight"] == nil)
        #expect(sanitized["vision_embedder.patch_dense.weight"] != nil)
        #expect(sanitized["embed_vision.embedding_projection.weight"] != nil)
        #expect(sanitized["embed_audio.embedding_projection.weight"] == nil)
    }

    @Test("Gemma4 Unified processor is registered")
    func processorIsRegistered() async throws {
        let data = Data(
            """
            {
              "processor_class": "Gemma4UnifiedProcessor",
              "image_token_id": 31,
              "boi_token_id": 28,
              "eoi_token_id": 29,
              "image_processor": {
                "patch_size": 2,
                "pooling_kernel_size": 2,
                "model_patch_size": 4,
                "max_soft_tokens": 4,
                "size": { "height": 8, "width": 8 }
              }
            }
            """.utf8)
        let processor = try await VLMProcessorTypeRegistry.shared.createModel(
            configuration: data,
            processorType: "Gemma4UnifiedProcessor",
            tokenizer: Gemma4UnifiedTestTokenizer())
        #expect(processor is Gemma4UnifiedProcessor)
    }

    @Test("Gemma4 Unified processor emits model patches and position ids")
    func processorPatchifiesImages() async throws {
        let data = Data(
            """
            {
              "processor_class": "Gemma4UnifiedProcessor",
              "image_token_id": 31,
              "boi_token_id": 28,
              "eoi_token_id": 29,
              "image_processor": {
                "patch_size": 2,
                "pooling_kernel_size": 2,
                "model_patch_size": 4,
                "max_soft_tokens": 4,
                "size": { "height": 8, "width": 8 }
              }
            }
            """.utf8)
        let config = try JSONDecoder.json5().decode(
            Gemma4UnifiedProcessorConfiguration.self, from: data)
        let processor = Gemma4UnifiedProcessor(config, tokenizer: Gemma4UnifiedTestTokenizer())
        let image = CIImage(color: .black).cropped(to: CGRect(x: 0, y: 0, width: 8, height: 8))

        let input = try await processor.prepare(
            input: UserInput(prompt: "describe", images: [.ciImage(image)]))

        #expect(input.image?.pixels.shape == [1, 4, 48])
        #expect(input.image?.positionIds?.shape == [1, 4, 2])
        #expect(input.text.tokens.asArray(Int32.self) == [28, 31, 31, 31, 31, 29, 2])
    }

    @Test("Gemma4 Unified prepare accepts images instead of rejecting media")
    func prepareAcceptsImages() throws {
        try MLXMetalTestLock.withLock {
            let model = Gemma4Unified(try decodeConfig(tinyTextJSON(vision: true, audio: false)))
            let cache = model.newCache(parameters: nil)
            let tokens = MLXArray([Int32(31), 31]).expandedDimensions(axis: 0)
            let pixels = MLXArray.zeros([1, 2, 48])
            let positions = MLXArray([Int32(0), 0, 1, 0]).reshaped(1, 2, 2)
            let input = LMInput(
                text: .init(tokens: tokens),
                image: .init(pixels: pixels, positionIds: positions))

            let result = try model.prepare(input, cache: cache, windowSize: nil)
            guard case .logits(let output) = result else {
                Issue.record("Expected Gemma4Unified.prepare to return logits for an image request")
                return
            }
            #expect(output.logits.shape.last == 32)
        }
    }

    @Test("Gemma4 Unified prepare accepts pre-encoded audio embeddings")
    func prepareAcceptsAudioEmbeddings() throws {
        try MLXMetalTestLock.withLock {
            let model = Gemma4Unified(try decodeConfig(tinyTextJSON(vision: false, audio: true)))
            let cache = model.newCache(parameters: nil)
            let tokens = MLXArray([Int32(30), 30]).expandedDimensions(axis: 0)
            let audio = LMInput.ProcessedAudio(
                waveform: MLXArray.zeros([1, 16]),
                preEncodedEmbedding: MLXArray.zeros([1, 2, 8]))
            let input = LMInput(text: .init(tokens: tokens), audio: audio)

            let result = try model.prepare(input, cache: cache, windowSize: nil)
            guard case .logits(let output) = result else {
                Issue.record("Expected Gemma4Unified.prepare to return logits for audio embeddings")
                return
            }
            #expect(output.logits.shape.last == 32)
        }
    }
}
