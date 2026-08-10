import Foundation
import MLX

public struct DFlashLinearArgs: @unchecked Sendable {
    public let target: any (HiddenStateCaptureModel & TokenEmbedderModel)
    public let drafter: DFlashDraftModel
    public let targetBlockIDs: [Int]
    public let maskTokenID: Int32
    public let inputIds: MLXArray
    public let maxNewTokens: Int
    public let stopTokenIDs: Set<Int32>
    public let temperature: Float

    public init(
        target: any(HiddenStateCaptureModel & TokenEmbedderModel),
        drafter: DFlashDraftModel,
        targetBlockIDs: [Int],
        maskTokenID: Int32,
        inputIds: MLXArray,
        maxNewTokens: Int,
        stopTokenIDs: Set<Int32> = [],
        temperature: Float = 0
    ) {
        self.target = target
        self.drafter = drafter
        self.targetBlockIDs = targetBlockIDs
        self.maskTokenID = maskTokenID
        self.inputIds = inputIds
        self.maxNewTokens = maxNewTokens
        self.stopTokenIDs = stopTokenIDs
        self.temperature = temperature
    }
}

public struct DFlashLinearResult: Sendable {
    public let tokenIds: [Int32]
    public let acceptanceLengths: [Int]
}

public enum SpecDecRuntimeLinear {
    public static func run(_ args: DFlashLinearArgs) -> DFlashLinearResult {
        let blockSize = validatedBlockSize(args)

        let promptLength = args.inputIds.dim(1)
        let limit = promptLength + args.maxNewTokens
        var tokenIDs = args.inputIds.asArray(Int32.self)
        var acceptedLengths: [Int] = []
        let layers = Set(args.targetBlockIDs)

        var (context, bonus) = initialState(args, layers: layers)
        tokenIDs.append(bonus)

        while tokenIDs.count < limit {
            if args.stopTokenIDs.contains(bonus) {
                break
            }

            let block = draftBlock(args, context: context, bonus: bonus, tokenCount: tokenIDs.count)

            let prefix = Array(tokenIDs.dropLast())
            let verificationIDs = prefix + block
            let verification = args.target(
                MLXArray(verificationIDs).reshaped(1, verificationIDs.count),
                cache: nil,
                captureLayerIDs: layers
            )
            MLX.eval(verification.logits)
            let blockStart = verificationIDs.count - blockSize
            let posterior = argMax(
                verification.logits[0..., blockStart..., 0...], axis: -1
            ).asType(.int32)
            MLX.eval(posterior)

            var accepted = 0
            while accepted < blockSize - 1, block[accepted + 1] == posterior[accepted].item(Int32.self) {
                accepted += 1
            }
            acceptedLengths.append(accepted)
            if accepted > 0 {
                tokenIDs.append(contentsOf: block[1 ... accepted])
            }
            bonus = posterior[accepted].item(Int32.self)
            tokenIDs.append(bonus)

            context = contextFeature(
                from: verification.capturedHiddenStates,
                args: args,
                length: prefix.count + accepted + 1
            )
            MLX.eval(context)
        }

        return .init(
            tokenIds: Array(tokenIDs.prefix(limit)),
            acceptanceLengths: acceptedLengths
        )
    }

    private static func validatedBlockSize(_ args: DFlashLinearArgs) -> Int {
        precondition(args.temperature == 0)
        precondition(args.inputIds.ndim == 2 && args.inputIds.dim(0) == 1)
        precondition(args.drafter.config.blockSize >= 2)
        precondition(args.maxNewTokens > 0)
        precondition(!args.targetBlockIDs.isEmpty)
        precondition(args.targetBlockIDs.allSatisfy { args.target.supportedCaptureLayerIDs.contains($0) })
        return args.drafter.config.blockSize
    }

    private static func sampleLastToken(_ logits: MLXArray) -> Int32 {
        let token = argMax(logits[0, logits.dim(1) - 1, 0...], axis: -1).asType(.int32)
        MLX.eval(token)
        return token.item(Int32.self)
    }

    private static func initialState(
        _ args: DFlashLinearArgs,
        layers: Set<Int>
    ) -> (context: MLXArray, bonus: Int32) {
        let prefill = args.target(args.inputIds, cache: nil, captureLayerIDs: layers)
        MLX.eval(prefill.logits)
        let context = contextFeature(from: prefill.capturedHiddenStates, args: args)
        MLX.eval(context)
        return (context, sampleLastToken(prefill.logits))
    }

    private static func draftBlock(
        _ args: DFlashLinearArgs,
        context: MLXArray,
        bonus: Int32,
        tokenCount: Int
    ) -> [Int32] {
        let blockSize = args.drafter.config.blockSize
        var block = Array(repeating: args.maskTokenID, count: blockSize)
        block[0] = bonus
        let blockIDs = MLXArray(block).reshaped(1, blockSize)
        let positions = MLXArray(
            (tokenCount - 1 ..< tokenCount - 1 + blockSize).map(Int32.init)
        ).reshaped(1, blockSize)
        let hidden = args.drafter(
            noiseEmbedding: args.target.embed(blockIDs),
            targetHidden: context,
            positionIDs: positions
        )
        let predictions = argMax(
            args.target.projectToLogits(hidden[0..., 1..., 0...]), axis: -1
        ).asType(.int32)
        MLX.eval(predictions)
        block.replaceSubrange(1..., with: predictions.asArray(Int32.self))
        return block
    }

    private static func contextFeature(
        from captured: [Int: MLXArray],
        args: DFlashLinearArgs,
        length: Int? = nil
    ) -> MLXArray {
        let context = extractContextFeature(
            captured: captured,
            targetLayerIDs: args.targetBlockIDs
        )
        guard let length else { return context }
        return context[0..., 0 ..< length, 0...]
    }
}
