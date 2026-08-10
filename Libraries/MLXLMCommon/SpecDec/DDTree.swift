public struct DDTreeCandidate: Sendable, Hashable {
    public let tokenID: Int32
    public let score: Float

    public init(tokenID: Int32, score: Float) {
        self.tokenID = tokenID
        self.score = score
    }
}

public struct DDTreeNode: Sendable, Hashable {
    public let tokenID: Int32
    public let parentID: Int?
    public let depth: Int
    public let score: Float

    public init(tokenID: Int32, parentID: Int?, depth: Int, score: Float) {
        self.tokenID = tokenID
        self.parentID = parentID
        self.depth = depth
        self.score = score
    }
}

public struct DDTree: Sendable {
    public let nodes: [DDTreeNode]

    public init(nodes: [DDTreeNode]) {
        precondition(!nodes.isEmpty)
        precondition(nodes[0].parentID == nil && nodes[0].depth == 0)
        for (nodeID, node) in nodes.enumerated().dropFirst() {
            guard let parentID = node.parentID else {
                preconditionFailure("Only the DDTree root may omit a parent")
            }
            precondition(parentID >= 0 && parentID < nodeID)
            precondition(node.depth == nodes[parentID].depth + 1)
        }
        self.nodes = nodes
    }

    public var rootTokenID: Int32 {
        nodes[0].tokenID
    }

    public func children(of nodeID: Int) -> [Int] {
        nodes.indices.filter { nodes[$0].parentID == nodeID }
    }

    public func tokenPath(to nodeID: Int) -> [Int32] {
        precondition(nodes.indices.contains(nodeID))
        var path: [Int32] = []
        var current: Int? = nodeID
        while let nodeID = current {
            path.append(nodes[nodeID].tokenID)
            current = nodes[nodeID].parentID
        }
        return path.reversed()
    }
}
