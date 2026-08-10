@testable import MLXLMCommon
import Testing

@Suite("DDTree", .serialized)
struct DDTreeTests {
    @Test("Tree builder distributes deterministic proposals across branches")
    func builderDistributesProposals() {
        let tree = DDTreeBuilder.build(
            rootTokenID: 1,
            blockSize: 4,
            branchingBudget: 5
        ) { path, _ in
            switch path {
            case [1]: [.init(tokenID: 2, score: 0.9), .init(tokenID: 9, score: 0.1)]
            case [1, 2]: [.init(tokenID: 3, score: 0.9), .init(tokenID: 7, score: 0.2)]
            case [1, 9]: [.init(tokenID: 8, score: 0.9)]
            default: []
            }
        }

        #expect(tree.nodes.map(\.tokenID) == [1, 2, 9, 3, 8, 7])
        #expect(tree.tokenPath(to: 4) == [1, 9, 8])
    }

    @Test("Compiler, verifier, and cache reconciliation preserve greedy path")
    func endToEndParity() throws {
        let tree = DDTreeBuilder.build(
            rootTokenID: 1,
            blockSize: 4,
            branchingBudget: 5
        ) { path, _ in
            switch path {
            case [1]: [.init(tokenID: 2, score: 1), .init(tokenID: 9, score: 0)]
            case [1, 2]: [.init(tokenID: 3, score: 1), .init(tokenID: 7, score: 0)]
            case [1, 9]: [.init(tokenID: 8, score: 1)]
            default: []
            }
        }
        let compilation = DDTreeCompiler.compile(tree, startingPosition: 11)
        let predicted = tree.nodes.map { token in
            switch token.tokenID {
            case 1: Int32(2)
            case 2: Int32(3)
            case 3: Int32(4)
            default: Int32(0)
            }
        }
        let verification = try DDTreeVerifier.verify(tree: tree, predictedTokenIDs: predicted)

        #expect(compilation.positionIDs == [11, 12, 12, 13, 13, 13])
        #expect(compilation.attentionMask[3] == [true, true, false, true, false, false])
        #expect(verification.acceptedTokenIDs + [verification.nextTokenID] == [2, 3, 4])

        let cache = KVCacheSimple()
        cache.offset = compilation.tree.nodes.count
        let trimmed = try DDTreeCacheIntegration.retainAcceptedPath(
            in: [cache],
            compilation: compilation,
            verification: verification
        )

        #expect(trimmed == 3)
        #expect(cache.offset == 3)
    }

    @Test("Cache forks isolate speculative tree verification")
    func cacheForkIsIndependent() throws {
        let cache = KVCacheSimple()
        cache.offset = 4
        let fork = DDTreeCacheIntegration.fork([cache])
        let forkedCache = try #require(fork[0] as? KVCacheSimple)
        forkedCache.offset = 7

        #expect(cache.offset == 4)
        #expect(forkedCache.offset == 7)
    }

    @Test("Verifier returns the root correction when no branch matches")
    func verifierReturnsRootCorrection() throws {
        let tree = DDTreeBuilder.build(
            rootTokenID: 1,
            blockSize: 2,
            branchingBudget: 1
        ) { _, _ in [.init(tokenID: 2, score: 1)] }

        let verification = try DDTreeVerifier.verify(tree: tree, predictedTokenIDs: [7, 0])

        #expect(verification.acceptedNodeIDs == [0])
        #expect(verification.acceptedTokenIDs.isEmpty)
        #expect(verification.nextTokenID == 7)
        #expect(verification.discardedNodeCount == 1)
    }
}
