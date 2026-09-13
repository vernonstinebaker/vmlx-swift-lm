import Foundation
import MLX

public enum SpecDecStrategyError: Error, LocalizedError {
    case unsupportedStrategy(String)
    case unsupportedTarget(String)
    case unsupportedInput
    case unsupportedSampling
    case incompatibleBlockSize(strategy: Int, drafter: Int)
    case invalidTargetLayerIDs
    case unsupportedTargetLayerID(Int)

    public var errorDescription: String? {
        switch self {
        case let .unsupportedStrategy(strategy): "Unsupported speculative strategy: \(strategy)"
        case let .unsupportedTarget(type): "\(type) does not expose DFlash target hooks"
        case .unsupportedInput: "DFlash strategies currently support text-only inputs"
        case .unsupportedSampling: "DFlash strategies currently require greedy sampling without logit penalties"
        case let .incompatibleBlockSize(strategy, drafter):
            "DFlash strategy block size \(strategy) does not match the drafter block size \(drafter)"
        case .invalidTargetLayerIDs:
            "DFlash drafters require unique, zero-based target layer IDs"
        case let .unsupportedTargetLayerID(layerID):
            "DFlash target layer \(layerID) is not available on the target model"
        }
    }
}

package enum SpecDecStrategyCachePolicy {
    package static func usesFullReprefill(
        cache: [KVCache]?,
        parameters: GenerateParameters
    ) throws -> Bool {
        guard cache == nil else {
            FileHandle.standardError.write(Data("[dflash2-probe][vMLX] reprefill=false (cache non-nil)\n".utf8))
            return false
        }
        let plan = try parameters.kvCachePlan()
        let reprefill = plan.configuration == nil
        FileHandle.standardError.write(
            Data("[dflash2-probe][vMLX] reprefill=\(reprefill) config=\(String(describing: plan.configuration))\n".utf8))
        return reprefill
    }
}

public struct SpecDecStrategyTokenIterator: TokenIteratorProtocol {
    public let runtimeKind: String
    public let maxTokens: Int?
    public private(set) var tokenCount = 0
    public let promptPrefillTime: TimeInterval = 0

    private let target: any (HiddenStateCaptureModel & TokenEmbedderModel)
    private let resolvedDrafter: ResolvedDrafter?
    private let resolvedDFlash2: ResolvedDFlash2Drafter?
    private let dflash2Target: (any (
        HiddenStateCaptureModel & TokenEmbedderModel & DFlash2VerifyRollbackModel
    ))?
    private let strategy: DraftStrategy
    private let stopTokenIDs: Set<Int32>
    private var inputIDs: [Int32]
    private var pending: [Int32] = []

    public init(
        input: LMInput,
        target: any LanguageModel,
        parameters: GenerateParameters,
        components: GenerationComponents = .init(),
        stopTokenIDs: Set<Int32> = [],
        resolver: SpecDecDrafterResolver = .shared
    ) throws {
        guard input.image == nil, input.video == nil, input.audio == nil else {
            throw SpecDecStrategyError.unsupportedInput
        }
        guard
            parameters.temperature == 0,
            parameters.processor() == nil,
            components.logitProcessor(parameters: parameters) == nil
        else {
            throw SpecDecStrategyError.unsupportedSampling
        }
        guard let strategy = parameters.draftStrategy, strategy.usesBlockDiffusion else {
            throw SpecDecStrategyError.unsupportedStrategy(Self.strategyName(parameters.draftStrategy))
        }
        guard let target = target as? any(HiddenStateCaptureModel & TokenEmbedderModel) else {
            throw SpecDecStrategyError.unsupportedTarget(String(describing: type(of: target)))
        }

        self.target = target
        if case .dflash2 = strategy {
            // DFlash 2 rollback on a hybrid target needs verify-input
            // stashing; without the conformance the cache cannot be trusted
            // past the first rejected block.
            guard target is DFlash2VerifyRollbackModel else {
                throw DFlash2RuntimeError.targetLacksPrefixCommitRecording
            }
            let resolved = try resolver.resolveDFlash2(strategy: strategy)
            resolvedDFlash2 = resolved
            resolvedDrafter = nil
            dflash2Target = target as? any (
                HiddenStateCaptureModel & TokenEmbedderModel & DFlash2VerifyRollbackModel
            )
            let configuredBlockSize = Self.blockSize(of: strategy)
            guard configuredBlockSize == resolved.blockSize else {
                throw SpecDecStrategyError.incompatibleBlockSize(
                    strategy: configuredBlockSize, drafter: resolved.blockSize
                )
            }
        } else {
            resolvedDFlash2 = nil
            dflash2Target = nil
            let resolved = try resolver.resolve(strategy: strategy)
            resolvedDrafter = resolved
            let configuredBlockSize = Self.blockSize(of: strategy)
            guard configuredBlockSize == resolved.model.config.blockSize else {
                throw SpecDecStrategyError.incompatibleBlockSize(
                    strategy: configuredBlockSize,
                    drafter: resolved.model.config.blockSize
                )
            }
        }
        let targetBlockIDs: [Int]
        if let dflash2 = resolvedDFlash2 {
            targetBlockIDs = dflash2.targetLayerIDs
        } else {
            // .dflash/.ddtree always resolve a v1 drafter; empty triggers the
            // invalidTargetLayerIDs guard below.
            targetBlockIDs = resolvedDrafter?.targetBlockIDs ?? []
        }
        guard
            !targetBlockIDs.isEmpty,
            targetBlockIDs.allSatisfy({ $0 >= 0 }),
            Set(targetBlockIDs).count == targetBlockIDs.count
        else {
            throw SpecDecStrategyError.invalidTargetLayerIDs
        }
        for layerID in targetBlockIDs where !target.supportedCaptureLayerIDs.contains(layerID) {
            throw SpecDecStrategyError.unsupportedTargetLayerID(layerID)
        }
        self.strategy = strategy
        self.stopTokenIDs = stopTokenIDs
        inputIDs = input.text.tokens.asArray(Int32.self)
        maxTokens = parameters.maxTokens
        runtimeKind = strategy.kindName
    }

    public mutating func next() -> Int? {
        guard maxTokens.map({ tokenCount < $0 }) ?? true else { return nil }
        if pending.isEmpty {
            pending = generateBlock()
        }
        guard let token = pending.first else { return nil }
        pending.removeFirst()
        inputIDs.append(token)
        tokenCount += 1
        return Int(token)
    }

    public mutating func discardGeneratedToken() {
        guard tokenCount > 0 else { return }
        inputIDs.removeLast()
        pending.removeAll()
        tokenCount -= 1
    }

    private mutating func generateBlock() -> [Int32] {
        let input = MLXArray(inputIDs).reshaped(1, inputIDs.count)
        let remainingTokens = maxTokens.map { $0 - tokenCount } ?? Self.blockSize(of: strategy)
        guard remainingTokens > 0 else { return [] }
        let blockSize = Self.blockSize(of: strategy)

        switch strategy {
        case .dflash:
            guard let resolved = resolvedDrafter else { return [] }
            let result = SpecDecRuntimeLinear.run(.init(
                target: target,
                drafter: resolved.model,
                targetBlockIDs: resolved.targetBlockIDs,
                maskTokenID: resolved.maskTokenID,
                inputIds: input,
                maxNewTokens: remainingTokens,
                stopTokenIDs: stopTokenIDs,
                temperature: 0
            ))
            return Array(result.tokenIds.dropFirst(inputIDs.count))

        case let .ddtree(_, branchingBudget, _):
            guard let resolved = resolvedDrafter else { return [] }
            let result = try? SpecDecRuntimeDDTree.run(.init(
                target: target,
                drafter: resolved.model,
                targetBlockIDs: resolved.targetBlockIDs,
                maskTokenID: resolved.maskTokenID,
                inputIDs: input,
                maxNewTokens: remainingTokens,
                stopTokenIDs: stopTokenIDs,
                branchingBudget: branchingBudget,
                blockSize: blockSize
            ))
            return result?.tokenIDs ?? []

        case .dflash2:
            guard let resolved = resolvedDFlash2, let dflash2Target else { return [] }
            // A runtime failure here (commit cannot be trusted, refuse
            // paths) ends the turn without emitting fabricated tokens.
            guard let result = try? SpecDecRuntimeDFlash2.run(.init(
                target: dflash2Target,
                drafter: resolved.model,
                targetLayerIDs: resolved.targetLayerIDs,
                maskTokenID: resolved.maskTokenID,
                inputIds: input,
                maxNewTokens: remainingTokens,
                stopTokenIDs: stopTokenIDs
            ))
            else { return [] }
            let pending = Array(result.tokenIds.dropFirst(inputIDs.count))
            FileHandle.standardError.write(Data(
                "[dflash2-probe][iterator] inputCount=\(inputIDs.count) resultRows=\(result.tokenIds.count) pending=\(pending.count) first8=\(Array(pending.prefix(8)))\n".utf8))
            return pending

        case .none, .autoregressive:
            return []
        }
    }

    private static func strategyName(_ strategy: DraftStrategy?) -> String {
        strategy?.kindName ?? "none"
    }

    private static func blockSize(of strategy: DraftStrategy) -> Int {
        switch strategy {
        case let .dflash(_, blockSize), let .ddtree(_, _, blockSize),
            let .dflash2(_, blockSize):
            blockSize
        case .none, .autoregressive: 0
        }
    }
}
