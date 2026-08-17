import MLX

public struct DDTreeArgs: @unchecked Sendable {
    public let target: any (HiddenStateCaptureModel & TokenEmbedderModel)
    public let drafter: DFlashDraftModel
    public let targetBlockIDs: [Int]
    public let maskTokenID: Int32
    public let inputIDs: MLXArray
    public let maxNewTokens: Int
    public let stopTokenIDs: Set<Int32>
    public let branchingBudget: Int
    public let blockSize: Int

    public init(
        target: any(HiddenStateCaptureModel & TokenEmbedderModel),
        drafter: DFlashDraftModel,
        targetBlockIDs: [Int],
        maskTokenID: Int32,
        inputIDs: MLXArray,
        maxNewTokens: Int,
        stopTokenIDs: Set<Int32> = [],
        branchingBudget: Int,
        blockSize: Int
    ) {
        self.target = target
        self.drafter = drafter
        self.targetBlockIDs = targetBlockIDs
        self.maskTokenID = maskTokenID
        self.inputIDs = inputIDs
        self.maxNewTokens = maxNewTokens
        self.stopTokenIDs = stopTokenIDs
        self.branchingBudget = branchingBudget
        self.blockSize = blockSize
    }
}

public struct DDTreeResult: Sendable {
    public let tokenIDs: [Int32]
}

public enum SpecDecRuntimeDDTree {
    public static func run(_ args: DDTreeArgs) throws -> DDTreeResult {
        precondition(args.inputIDs.ndim == 2 && args.inputIDs.dim(0) == 1)
        precondition(args.maxNewTokens > 0)
        precondition(args.blockSize >= 2)
        precondition(args.branchingBudget > 0)
        precondition(args.blockSize == args.drafter.config.blockSize)

        let draft = SpecDecRuntimeLinear.run(.init(
            target: args.target,
            drafter: args.drafter,
            targetBlockIDs: args.targetBlockIDs,
            maskTokenID: args.maskTokenID,
            inputIds: args.inputIDs,
            maxNewTokens: args.maxNewTokens,
            stopTokenIDs: args.stopTokenIDs,
            temperature: 0
        ))
        let promptLength = args.inputIDs.dim(1)
        let proposals = Array(draft.tokenIds.dropFirst(promptLength))
        guard let rootTokenID = proposals.first else {
            return .init(tokenIDs: [])
        }

        let cache = try args.target.newCache(parameters: nil)
        let prefill = args.target(args.inputIDs, cache: cache, captureLayerIDs: [])
        let rootPrediction = argMax(
            prefill.logits[0, prefill.logits.dim(1) - 1, 0...], axis: -1
        ).asType(.int32)
        eval(rootPrediction)

        let tree = DDTreeBuilder.build(
            rootTokenID: rootTokenID,
            blockSize: args.blockSize,
            branchingBudget: args.branchingBudget
        ) { path, _ in
            let proposalIndex = path.count
            guard proposals.indices.contains(proposalIndex) else { return [] }
            return [.init(tokenID: proposals[proposalIndex], score: 1)]
        }
        let verification = try DDTreeBranchVerifier.verify(
            target: args.target,
            tree: tree,
            cache: cache,
            rootPrediction: rootPrediction.item(Int32.self)
        )

        var tokenIDs = verification.verification.acceptedNodeIDs.map { tree.nodes[$0].tokenID }
        tokenIDs.append(verification.verification.nextTokenID)
        return .init(tokenIDs: Array(tokenIDs.prefix(args.maxNewTokens)))
    }
}
