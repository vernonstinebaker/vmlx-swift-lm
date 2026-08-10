public enum DDTreeBuilder {
    public static func build(
        rootTokenID: Int32,
        blockSize: Int,
        branchingBudget: Int,
        propose: ([Int32], Int) -> [DDTreeCandidate]
    ) -> DDTree {
        precondition(blockSize >= 2)
        precondition(branchingBudget > 0)

        var nodes = [DDTreeNode(tokenID: rootTokenID, parentID: nil, depth: 0, score: 0)]
        var frontier = [(nodeID: 0, path: [rootTokenID])]

        for depth in 1 ..< blockSize {
            let remaining = branchingBudget - (nodes.count - 1)
            guard remaining > 0, !frontier.isEmpty else { break }

            let proposals = frontier.map { entry in
                (
                    nodeID: entry.nodeID,
                    path: entry.path,
                    candidates: ranked(propose(entry.path, remaining))
                )
            }
            let rankCount = proposals.map(\.candidates.count).max() ?? 0
            var nextFrontier: [(nodeID: Int, path: [Int32])] = []

            for rank in 0 ..< rankCount {
                for proposal in proposals where proposal.candidates.indices.contains(rank) {
                    guard nodes.count - 1 < branchingBudget else {
                        return DDTree(nodes: nodes)
                    }
                    let candidate = proposal.candidates[rank]
                    let nodeID = nodes.count
                    nodes.append(.init(
                        tokenID: candidate.tokenID,
                        parentID: proposal.nodeID,
                        depth: depth,
                        score: candidate.score
                    ))
                    nextFrontier.append((nodeID, proposal.path + [candidate.tokenID]))
                }
            }
            frontier = nextFrontier
        }

        return DDTree(nodes: nodes)
    }

    private static func ranked(_ candidates: [DDTreeCandidate]) -> [DDTreeCandidate] {
        var seen = Set<Int32>()
        return candidates
            .sorted {
                if $0.score == $1.score {
                    return $0.tokenID < $1.tokenID
                }
                return $0.score > $1.score
            }
            .filter { seen.insert($0.tokenID).inserted }
    }
}
