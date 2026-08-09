import MLX
import MLXNN

public final class BatchKVCache: BaseKVCache {
    private let slotCaches: [KVCache]
    public let batchSize: Int
    public private(set) var offsetArray: MLXArray

    public init(slotCaches: [KVCache]) {
        precondition(!slotCaches.isEmpty)
        self.slotCaches = slotCaches
        batchSize = slotCaches.count
        offsetArray = MLXArray(slotCaches.map { Int32($0.offset) })
        super.init()
        offset = slotCaches.map(\.offset).max() ?? 0
    }

    public override var ropeOffset: RoPEOffset {
        .batch(offsetArray)
    }

    public override func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        precondition(keys.dim(0) == batchSize)
        var allKeys: [MLXArray] = []
        var allValues: [MLXArray] = []
        for index in 0 ..< batchSize {
            let updated = slotCaches[index].update(
                keys: keys[index ..< index + 1], values: values[index ..< index + 1])
            allKeys.append(updated.0)
            allValues.append(updated.1)
        }
        offsetArray = MLXArray(slotCaches.map { Int32($0.offset) })
        offset = slotCaches.map(\.offset).max() ?? 0
        return (padAndConcatenate(allKeys), padAndConcatenate(allValues))
    }

    public override func makeMask(
        n: Int, windowSize: Int?, returnArray _: Bool
    ) -> MLXFast.ScaledDotProductAttentionMaskMode {
        let offsets = slotCaches.map(\.offset)
        let keyLengths = slotCaches.map { cache in
            min(cache.offset + n, cache.maxSize ?? .max)
        }
        return .array(createBatchCausalMask(
            queryLen: n, offsets: offsets, effectiveKeyLens: keyLengths, windowSize: windowSize))
    }

    public override var state: [MLXArray] {
        get { [] }
        set {}
    }

    public override var metaState: [String] {
        get { [""] }
        set {}
    }

    public override func copy() -> any KVCache {
        fatalError("BatchKVCache is a transient cache view")
    }

    private func padAndConcatenate(_ arrays: [MLXArray]) -> MLXArray {
        let maxLength = arrays.map { $0.dim(2) }.max() ?? 0
        return concatenated(arrays.map { array in
            guard array.dim(2) < maxLength else { return array }
            var shape = array.shape
            shape[2] = maxLength - array.dim(2)
            return concatenated([array, MLXArray.zeros(shape, dtype: array.dtype)], axis: 2)
        }, axis: 0)
    }
}
