import Foundation

public enum DraftStrategy: @unchecked Sendable {
    case none
    case autoregressive(draftModel: any LanguageModel, numDraftTokens: Int)
    case dflash(drafterPath: URL, blockSize: Int)
    case ddtree(drafterPath: URL, branchingBudget: Int, blockSize: Int)
    case dflash2(drafterPath: URL, blockSize: Int)

    public var kindName: String {
        switch self {
        case .none: "none"
        case .autoregressive: "autoregressive"
        case .dflash: "dflash"
        case .ddtree: "ddtree"
        case .dflash2: "dflash2"
        }
    }

    public var usesBlockDiffusion: Bool {
        switch self {
        case .dflash, .ddtree, .dflash2: true
        case .none, .autoregressive: false
        }
    }
}
