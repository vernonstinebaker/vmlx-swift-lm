import Foundation
import MLXLMCommon
@testable import MLXVLM
import Testing

@Suite("Restored model registrations")
struct RecoveredModelRegistrationTests {
    @Test("Gemma 4 Unified remains a supported VLM architecture")
    func gemma4UnifiedIsRegistered() {
        #expect(VLMTypeRegistry.supportedModelTypes.contains("gemma4_unified"))
    }

    @Test("Gemma 4 Unified accepts its native configuration")
    func gemma4UnifiedDecodesItsNativeConfiguration() throws {
        let configuration = try JSONDecoder().decode(
            Gemma4UnifiedConfiguration.self,
            from: Data("""
            {
              "model_type": "gemma4_unified",
              "text_config": {},
              "vision_config": {}
            }
            """.utf8))

        #expect(configuration.modelType == "gemma4_unified")
    }

    @Test("Muse Glimmer remains a supported VLM architecture")
    func museGlimmerIsRegistered() {
        #expect(VLMTypeRegistry.supportedModelTypes.contains("muse_glimmer"))
    }
}
