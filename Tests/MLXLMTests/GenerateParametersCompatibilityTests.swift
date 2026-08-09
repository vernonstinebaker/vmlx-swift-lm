import Foundation
import Testing

@testable import MLXLMCommon

@Suite("Generate parameter compatibility")
struct GenerateParametersCompatibilityTests {
    @Test("Batch and stop-string preferences preserve caller values")
    func preservesBatchAndStopStringPreferences() {
        let parameters = GenerateParameters(
            enableCompiledBatchDecode: true,
            compiledBatchBuckets: [1, 4],
            draftStrategy: .dflash(
                drafterPath: URL(fileURLWithPath: "/tmp/drafter"), blockSize: 8),
            extraStopStrings: ["<stop>"])

        #expect(parameters.enableCompiledBatchDecode)
        #expect(parameters.compiledBatchBuckets == [1, 4])
        #expect(parameters.draftStrategy?.kindName == "dflash")
        #expect(parameters.draftStrategy?.usesBlockDiffusion == true)
        #expect(parameters.extraStopStrings == ["<stop>"])
    }
}
