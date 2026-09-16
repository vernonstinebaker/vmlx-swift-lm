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

    @Test("Gemma 4 Unified accepts its omitted vision-token count")
    func gemma4UnifiedDefaultsVisionTokenCount() throws {
        let configuration = try JSONDecoder().decode(
            Gemma4Configuration.self,
            from: Data("""
            {
              "model_type": "gemma4_unified",
              "text_config": {},
              "vision_config": {}
            }
            """.utf8))

        #expect(configuration.visionSoftTokensPerImage == 280)
    }

    @Test("Muse Glimmer remains a supported VLM architecture")
    func museGlimmerIsRegistered() {
        #expect(VLMTypeRegistry.supportedModelTypes.contains("muse_glimmer"))
    }
}
