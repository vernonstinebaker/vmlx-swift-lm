import Foundation
import MLX

public struct BatchRequestID: Hashable, Sendable, CustomStringConvertible {
    private let value: UUID

    public init() {
        value = UUID()
    }

    public var description: String {
        String(value.uuidString.prefix(8)).lowercased()
    }
}

public enum BatchGeneration: Sendable {
    case token(Int)
    case info(GenerateCompletionInfo)
}

struct BatchPendingRequest {
    let id = BatchRequestID()
    let input: LMInput
    let parameters: GenerateParameters
    let continuation: AsyncStream<BatchGeneration>.Continuation
}

enum BatchSlotPhase {
    case prefill
    case decode
}

struct BatchSlot {
    let id: BatchRequestID
    let input: LMInput
    let continuation: AsyncStream<BatchGeneration>.Continuation
    let sampler: LogitSampler
    var processor: LogitProcessor?
    let cache: [KVCache]
    let maxTokens: Int?
    let promptTokenCount: Int
    let prefillStartedAt = Date()
    var decodeStartedAt: Date?
    var nextToken: MLXArray?
    var phase: BatchSlotPhase = .prefill
    var generatedTokenCount = 0
    var isFinished = false

    init(request: BatchPendingRequest, cache: [KVCache]) {
        id = request.id
        input = request.input
        continuation = request.continuation
        sampler = request.parameters.sampler()
        processor = request.parameters.processor()
        self.cache = cache
        maxTokens = request.parameters.maxTokens
        promptTokenCount = request.input.text.tokens.size
    }

    mutating func sample(_ logits: MLXArray) -> MLXArray {
        guard var processor else {
            return sampler.sample(logits: logits)
        }
        let token = sampler.sample(logits: processor.process(logits: logits))
        processor.didSample(token: token)
        self.processor = processor
        return token
    }
}
