public enum DDTreeVerificationError: Error, Equatable {
    case invalidPredictionCount(expected: Int, actual: Int)
}

public struct DDTreeVerification: Sendable, Equatable {
    public let acceptedNodeIDs: [Int]
    public let acceptedTokenIDs: [Int32]
    public let nextTokenID: Int32
    public let treeNodeCount: Int

    public var discardedNodeCount: Int {
        treeNodeCount - acceptedNodeIDs.count
    }
}

public enum DDTreeVerifier {
    public static func verify(
        tree: DDTree,
        rootPrediction: Int32,
        predictedTokenIDs: [Int32]
    ) throws -> DDTreeVerification {
        guard predictedTokenIDs.count == tree.nodes.count else {
            throw DDTreeVerificationError.invalidPredictionCount(
                expected: tree.nodes.count,
                actual: predictedTokenIDs.count
            )
        }
        guard rootPrediction == tree.rootTokenID else {
            return .init(
                acceptedNodeIDs: [],
                acceptedTokenIDs: [],
                nextTokenID: rootPrediction,
                treeNodeCount: tree.nodes.count
            )
        }
        return try verify(tree: tree, predictedTokenIDs: predictedTokenIDs)
    }

    public static func verify(
        tree: DDTree,
        predictedTokenIDs: [Int32]
    ) throws -> DDTreeVerification {
        guard predictedTokenIDs.count == tree.nodes.count else {
            throw DDTreeVerificationError.invalidPredictionCount(
                expected: tree.nodes.count,
                actual: predictedTokenIDs.count
            )
        }

        var acceptedNodeIDs = [0]
        var currentNodeID = 0
        while let childNodeID = tree.children(of: currentNodeID).first(where: {
            tree.nodes[$0].tokenID == predictedTokenIDs[currentNodeID]
        }) {
            acceptedNodeIDs.append(childNodeID)
            currentNodeID = childNodeID
        }

        return .init(
            acceptedNodeIDs: acceptedNodeIDs,
            acceptedTokenIDs: acceptedNodeIDs.dropFirst().map { tree.nodes[$0].tokenID },
            nextTokenID: predictedTokenIDs[currentNodeID],
            treeNodeCount: tree.nodes.count
        )
    }
}
