import MLX

public enum DDTreeBranchVerifierError: Error, Equatable {
    case emptyCache
    case unexpectedPosition(nodeID: Int, expected: Int, actual: Int)
}

public struct DDTreeBranchVerification {
    public let verification: DDTreeVerification
    public let predictedTokenIDs: [Int32]
    public let positionIDs: [Int]
    public let cache: [KVCache]
}

/// Correctness-first verifier for trees whose target does not support a safe
/// one-pass tree forward. Each node gets an independent cache fork, so this
/// deliberately does not batch or share sibling execution.
public enum DDTreeBranchVerifier {
    public static func verify<Target: LanguageModel>(
        target: Target,
        tree: DDTree,
        cache: [KVCache],
        rootPrediction: Int32
    ) throws -> DDTreeBranchVerification {
        let startingPosition = try positionOffset(of: cache)
        let compilation = DDTreeCompiler.compile(tree, startingPosition: startingPosition)
        var nodeCaches: [[KVCache]] = []
        var predictedTokenIDs: [Int32] = []
        var positionIDs: [Int] = []

        nodeCaches.reserveCapacity(tree.nodes.count)
        predictedTokenIDs.reserveCapacity(tree.nodes.count)
        positionIDs.reserveCapacity(tree.nodes.count)

        for (nodeID, node) in tree.nodes.enumerated() {
            let nodeCache = DDTreeCacheIntegration.fork(
                node.parentID.map { nodeCaches[$0] } ?? cache
            )
            let position = try positionOffset(of: nodeCache)
            let expectedPosition = Int(compilation.positionIDs[nodeID])
            guard position == expectedPosition else {
                throw DDTreeBranchVerifierError.unexpectedPosition(
                    nodeID: nodeID,
                    expected: expectedPosition,
                    actual: position
                )
            }

            let logits = target(
                MLXArray([node.tokenID]).reshaped(1, 1),
                cache: nodeCache
            )
            let predicted = argMax(logits[0, logits.dim(1) - 1, 0...], axis: -1).asType(.int32)
            eval(predicted)

            nodeCaches.append(nodeCache)
            predictedTokenIDs.append(predicted.item(Int32.self))
            positionIDs.append(position)
        }

        let verification = try DDTreeVerifier.verify(
            tree: tree,
            rootPrediction: rootPrediction,
            predictedTokenIDs: predictedTokenIDs
        )
        let acceptedCache: [KVCache]
        if let nodeID = verification.acceptedNodeIDs.last {
            acceptedCache = nodeCaches[nodeID]
        } else {
            acceptedCache = DDTreeCacheIntegration.fork(cache)
        }

        return .init(
            verification: verification,
            predictedTokenIDs: predictedTokenIDs,
            positionIDs: positionIDs,
            cache: acceptedCache
        )
    }

    private static func positionOffset(of cache: [KVCache]) throws -> Int {
        guard let position = cache.map(\.offset).max() else {
            throw DDTreeBranchVerifierError.emptyCache
        }
        return position
    }
}
