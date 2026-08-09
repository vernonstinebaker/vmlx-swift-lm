import MLX

public func createBatchCausalMask(
    queryLen: Int,
    offsets: [Int],
    effectiveKeyLens: [Int]? = nil,
    windowSize: Int? = nil
) -> MLXArray {
    precondition(!offsets.isEmpty)
    let keyLengths = effectiveKeyLens ?? offsets.map { $0 + queryLen }
    precondition(keyLengths.count == offsets.count)
    let maxLength = keyLengths.max() ?? 0
    let keyIndices = MLXArray(Int32(0) ..< Int32(maxLength)).reshaped(1, maxLength)

    return concatenated(zip(offsets, keyLengths).map { offset, keyLength in
        let mask: MLXArray
        if keyLength < offset + queryLen {
            mask = MLX.broadcast(
                (keyIndices .< Int32(keyLength)).reshaped(1, maxLength),
                to: [queryLen, maxLength])
        } else {
            let queryIndices = (MLXArray(Int32(0) ..< Int32(queryLen)) + Int32(offset)).reshaped(queryLen, 1)
            var causal = queryIndices .>= keyIndices
            if let windowSize {
                causal = causal & (keyIndices .>= queryIndices - Int32(windowSize - 1))
            }
            mask = causal & (keyIndices .< Int32(keyLength))
        }
        return mask.reshaped(1, 1, queryLen, maxLength)
    }, axis: 0)
}
