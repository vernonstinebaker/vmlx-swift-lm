import Foundation

public enum DraftStrategy: @unchecked Sendable {
    case none
    case autoregressive(draftModel: any LanguageModel, numDraftTokens: Int)
    case dflash(drafterPath: URL, blockSize: Int)
    case ddtree(drafterPath: URL, branchingBudget: Int, blockSize: Int)

    public var kindName: String {
        switch self {
        case .none: "none"
        case .autoregressive: "autoregressive"
        case .dflash: "dflash"
        case .ddtree: "ddtree"
        }
    }

    public var usesBlockDiffusion: Bool {
        switch self {
        case .dflash, .ddtree: true
        case .none, .autoregressive: false
        }
    }
}
