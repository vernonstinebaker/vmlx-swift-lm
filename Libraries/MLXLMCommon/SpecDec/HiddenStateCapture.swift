import MLX

public protocol HiddenStateCaptureModel: LanguageModel {
    var supportedCaptureLayerIDs: Range<Int> { get }

    func callAsFunction(
        _ inputs: MLXArray,
        cache: [KVCache]?,
        captureLayerIDs: Set<Int>
    ) -> (logits: MLXArray, capturedHiddenStates: [Int: MLXArray])
}

public extension HiddenStateCaptureModel {
    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        callAsFunction(inputs, cache: cache, captureLayerIDs: []).logits
    }
}

public func extractContextFeature(
    captured: [Int: MLXArray],
    targetLayerIDs: [Int]
) -> MLXArray {
    precondition(!targetLayerIDs.isEmpty)
    return concatenated(targetLayerIDs.map { layerID in
        guard let hidden = captured[layerID] else {
            preconditionFailure("Missing hidden state for DFlash target layer \(layerID)")
        }
        return hidden
    }, axis: -1)
}
