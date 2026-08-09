import MLX

public final class BatchArraysCache: MambaCache {
    private let slotCaches: [ArraysCache]
    private var offsets: [Int]
    public let sequenceCount: Int
    private let stateSlotCount: Int
    public private(set) var offsetArray: MLXArray

    public init(slotCaches: [ArraysCache]) {
        precondition(!slotCaches.isEmpty)
        let stateSlotCount = slotCaches[0].slotCount
        precondition(slotCaches.allSatisfy { $0.slotCount == stateSlotCount })
        self.slotCaches = slotCaches
        self.stateSlotCount = stateSlotCount
        sequenceCount = slotCaches.count
        offsets = slotCaches.map(\.offset)
        offsetArray = MLXArray(offsets.map(Int32.init))
        super.init()
        offset = offsets.max() ?? 0
        for slot in 0 ..< stateSlotCount {
            let states = slotCaches.compactMap { $0[slot] }
            if states.count == sequenceCount {
                self[slot] = concatenated(states, axis: 0)
            }
        }
    }

    public override var ropeOffset: RoPEOffset {
        .batch(offsetArray)
    }

    public func splitBack() {
        for slot in 0 ..< stateSlotCount {
            guard let state = self[slot] else { continue }
            for (index, cache) in slotCaches.enumerated() {
                cache[slot] = state[index ..< index + 1]
                cache.offset = offsets[index]
            }
        }
    }

    public func advance(by count: Int) {
        offsets = offsets.map { $0 + count }
        offsetArray = MLXArray(offsets.map(Int32.init))
        offset = offsets.max() ?? 0
    }
}
