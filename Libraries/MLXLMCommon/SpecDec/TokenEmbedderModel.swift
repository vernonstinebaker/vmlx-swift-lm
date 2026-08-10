import MLX

public protocol TokenEmbedderModel: LanguageModel {
    func embed(_ tokenIDs: MLXArray) -> MLXArray
    func projectToLogits(_ hidden: MLXArray) -> MLXArray
}
