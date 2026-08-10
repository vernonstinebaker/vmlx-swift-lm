import MLX

public struct DDTreeCompilation: @unchecked Sendable {
    public let tree: DDTree
    public let tokenIDs: [Int32]
    public let positionIDs: [Int32]
    public let attentionMask: [[Bool]]

    public var inputIDs: MLXArray {
        MLXArray(tokenIDs)
    }

    public var batchedInputIDs: MLXArray {
        inputIDs.reshaped(1, tokenIDs.count)
    }

    public var attentionMaskArray: MLXArray {
        MLXArray(attentionMask.flatMap { $0 })
            .reshaped(tokenIDs.count, tokenIDs.count)
    }
}

public enum DDTreeCompiler {
    public static func compile(_ tree: DDTree, startingPosition: Int) -> DDTreeCompilation {
        precondition(startingPosition >= 0)
        let nodeIDs = tree.nodes.indices
        let attentionMask = nodeIDs.map { queryNodeID in
            let ancestors = ancestry(of: queryNodeID, in: tree)
            return nodeIDs.map(ancestors.contains)
        }
        return .init(
            tree: tree,
            tokenIDs: tree.nodes.map(\.tokenID),
            positionIDs: tree.nodes.map { Int32(startingPosition + $0.depth) },
            attentionMask: attentionMask
        )
    }

    private static func ancestry(of nodeID: Int, in tree: DDTree) -> Set<Int> {
        var ancestors = Set<Int>()
        var current: Int? = nodeID
        while let nodeID = current {
            ancestors.insert(nodeID)
            current = tree.nodes[nodeID].parentID
        }
        return ancestors
    }
}
