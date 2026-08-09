import MLX

public final class CompilableKVCache: BaseKVCache {
    public var keys: MLXArray?
    public var values: MLXArray?
    public var offsetArray: MLXArray
    public let maxLength: Int
    public var step: Int

    private lazy var maskIndices = MLXArray(Int32(0) ..< Int32(maxLength))

    public init(maxLength: Int = 4096, step: Int = 256) {
        self.maxLength = maxLength
        self.step = step
        offsetArray = MLXArray([Int32(0)])
        super.init()
    }

    public convenience init(from cache: KVCache, maxLength: Int = 4096) {
        self.init(maxLength: maxLength)
        let state = cache.state
        guard state.count >= 2 else { return }

        let existingKeys = state[0]
        let existingValues = state[1]
        let length = existingKeys.dim(2)
        precondition(length <= maxLength, "Compiled cache length exceeds its fixed capacity")

        keys = MLXArray.zeros(
            [existingKeys.dim(0), existingKeys.dim(1), maxLength, existingKeys.dim(3)],
            dtype: existingKeys.dtype)
        values = MLXArray.zeros(
            [existingValues.dim(0), existingValues.dim(1), maxLength, existingValues.dim(3)],
            dtype: existingValues.dtype)
        keys![.ellipsis, ..<length, 0...] = existingKeys
        values![.ellipsis, ..<length, 0...] = existingValues
        offsetArray = MLXArray([Int32(length)])
    }

    public override var offset: Int {
        get { offsetArray[0].item(Int.self) }
        set { offsetArray = MLXArray([Int32(newValue)]) }
    }

    public override func innerState() -> [MLXArray] {
        guard let keys, let values else { return [offsetArray] }
        return [keys, values, offsetArray]
    }

    public override func update(keys newKeys: MLXArray, values newValues: MLXArray) -> (MLXArray, MLXArray) {
        if keys == nil {
            keys = MLXArray.zeros(
                [newKeys.dim(0), newKeys.dim(1), maxLength, newKeys.dim(3)],
                dtype: newKeys.dtype)
            values = MLXArray.zeros(
                [newValues.dim(0), newValues.dim(1), maxLength, newValues.dim(3)],
                dtype: newValues.dtype)
        }

        let previousOffset = offsetArray
        let nextOffset = previousOffset + MLXArray([Int32(newKeys.dim(2))])
        keys!._updateInternal(
            dynamicSliceUpdate(keys!, update: newKeys, start: previousOffset, axes: [2]))
        values!._updateInternal(
            dynamicSliceUpdate(values!, update: newValues, start: previousOffset, axes: [2]))
        offsetArray._updateInternal(nextOffset)
        return (keys!, values!)
    }

    public override func makeMask(
        n: Int,
        windowSize: Int?,
        returnArray _: Bool
    ) -> MLXFast.ScaledDotProductAttentionMaskMode {
        let queryIndices: MLXArray
        if n == 1 {
            queryIndices = offsetArray.reshaped(1, 1)
        } else {
            queryIndices = (MLXArray(Int32(0) ..< Int32(n)) + offsetArray).reshaped(n, 1)
        }
        let keyIndices = maskIndices.reshaped(1, maxLength)
        var mask = queryIndices .>= keyIndices
        if let windowSize {
            mask = mask & (keyIndices .>= queryIndices - Int32(windowSize - 1))
        }
        return .array(mask)
    }

    public override var state: [MLXArray] {
        get {
            guard let keys, let values else { return [] }
            let length = offset
            return [keys[.ellipsis, ..<length, 0...], values[.ellipsis, ..<length, 0...]]
        }
        set {
            guard newValue.count == 2 else { return }
            let length = newValue[0].dim(2)
            precondition(length <= maxLength, "Compiled cache length exceeds its fixed capacity")
            keys = MLXArray.zeros(
                [newValue[0].dim(0), newValue[0].dim(1), maxLength, newValue[0].dim(3)],
                dtype: newValue[0].dtype)
            values = MLXArray.zeros(
                [newValue[1].dim(0), newValue[1].dim(1), maxLength, newValue[1].dim(3)],
                dtype: newValue[1].dtype)
            keys![.ellipsis, ..<length, 0...] = newValue[0]
            values![.ellipsis, ..<length, 0...] = newValue[1]
            offsetArray = MLXArray([Int32(length)])
        }
    }

    public override var isTrimmable: Bool { true }

    @discardableResult
    public override func trim(_ n: Int) -> Int {
        let trimmed = min(offset, n)
        offset -= trimmed
        return trimmed
    }

    public override func copy() -> any KVCache {
        let copy = CompilableKVCache(maxLength: maxLength, step: step)
        copy.keys = keys
        copy.values = values
        copy.offsetArray = offsetArray
        return copy
    }
}

extension CompilableKVCache: Updatable {}
