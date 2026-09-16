import MLXLMCommon
@testable import MLXVLM
import Testing

@Suite("Restored model registrations")
struct RecoveredModelRegistrationTests {
    @Test("Gemma 4 Unified remains a supported VLM architecture")
    func gemma4UnifiedIsRegistered() {
        #expect(VLMTypeRegistry.supportedModelTypes.contains("gemma4_unified"))
    }

    @Test("Muse Glimmer remains a supported VLM architecture")
    func museGlimmerIsRegistered() {
        #expect(VLMTypeRegistry.supportedModelTypes.contains("muse_glimmer"))
    }
}
