import Foundation
import MLXLMCommon

public typealias ModelContainer = EmbedderModelContainer

public func loadModelContainer(
    from directory: URL,
    using tokenizerLoader: any TokenizerLoader
) async throws -> ModelContainer {
    try await EmbedderModelFactory.shared.loadContainer(
        from: directory, using: tokenizerLoader)
}
