import Foundation
import MLX

public enum SpecDecStrategyError: Error, LocalizedError {
    case unsupportedStrategy(String)
    case unsupportedTarget(String)
    case unsupportedInput
    case unsupportedSampling

    public var errorDescription: String? {
        switch self {
        case let .unsupportedStrategy(strategy): "Unsupported speculative strategy: \(strategy)"
        case let .unsupportedTarget(type): "\(type) does not expose DFlash target hooks"
        case .unsupportedInput: "DFlash strategies currently support text-only inputs"
        case .unsupportedSampling: "DFlash strategies currently require greedy sampling without logit penalties"
        }
    }
}

public struct SpecDecStrategyTokenIterator: TokenIteratorProtocol {
    public let runtimeKind: String
    public let maxTokens: Int?
    public private(set) var tokenCount = 0
    public let promptPrefillTime: TimeInterval = 0

    private let target: any (HiddenStateCaptureModel & TokenEmbedderModel)
    private let resolvedDrafter: ResolvedDrafter
    private let strategy: DraftStrategy
    private var inputIDs: [Int32]
    private var pending: [Int32] = []

    public init(
        input: LMInput,
        target: any LanguageModel,
        parameters: GenerateParameters,
        resolver: SpecDecDrafterResolver = .shared
    ) throws {
        guard input.image == nil, input.video == nil, input.audio == nil else {
            throw SpecDecStrategyError.unsupportedInput
        }
        guard parameters.temperature == 0, parameters.processor() == nil else {
            throw SpecDecStrategyError.unsupportedSampling
        }
        guard let strategy = parameters.draftStrategy, strategy.usesBlockDiffusion else {
            throw SpecDecStrategyError.unsupportedStrategy(Self.strategyName(parameters.draftStrategy))
        }
        guard let target = target as? any(HiddenStateCaptureModel & TokenEmbedderModel) else {
            throw SpecDecStrategyError.unsupportedTarget(String(describing: type(of: target)))
        }

        self.target = target
        resolvedDrafter = try resolver.resolve(strategy: strategy)
        self.strategy = strategy
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
        let blockSize: Int
        switch strategy {
        case let .dflash(_, configuredBlockSize), let .ddtree(_, _, configuredBlockSize):
            blockSize = configuredBlockSize
        case .none, .autoregressive:
            return []
        }

        switch strategy {
        case .dflash:
            let result = SpecDecRuntimeLinear.run(.init(
                target: target,
                drafter: resolvedDrafter.model,
                targetBlockIDs: resolvedDrafter.targetBlockIDs,
                maskTokenID: resolvedDrafter.maskTokenID,
                inputIds: input,
                maxNewTokens: blockSize,
                temperature: 0
            ))
            return Array(result.tokenIds.dropFirst(inputIDs.count))

        case let .ddtree(_, branchingBudget, _):
            let result = try? SpecDecRuntimeDDTree.run(.init(
                target: target,
                drafter: resolvedDrafter.model,
                targetBlockIDs: resolvedDrafter.targetBlockIDs,
                maskTokenID: resolvedDrafter.maskTokenID,
                inputIDs: input,
                maxNewTokens: blockSize,
                branchingBudget: branchingBudget,
                blockSize: blockSize
            ))
            return result?.tokenIDs ?? []

        case .none, .autoregressive:
            return []
        }
    }

    private static func strategyName(_ strategy: DraftStrategy?) -> String {
        strategy?.kindName ?? "none"
    }
}
