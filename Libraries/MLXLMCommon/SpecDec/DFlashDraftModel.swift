import MLX
import MLXNN

public struct DFlashInnerConfig: Decodable, Sendable {
    public let maskTokenID: Int
    public let targetLayerIDs: [Int]
    public let blockSize: Int?

    enum CodingKeys: String, CodingKey {
        case maskTokenID = "mask_token_id"
        case targetLayerIDs = "target_layer_ids"
        case blockSize = "block_size"
    }
}

private struct DFlashRoPEParameters: Decodable {
    let ropeTheta: Float?

    enum CodingKeys: String, CodingKey {
        case ropeTheta = "rope_theta"
    }
}

public struct DFlashDrafterConfiguration: Decodable, Sendable {
    public let hiddenSize: Int
    public let numHiddenLayers: Int
    public let numAttentionHeads: Int
    public let numKeyValueHeads: Int
    public let headDim: Int
    public let intermediateSize: Int
    public let rmsNormEps: Float
    public let ropeTheta: Float
    public let attentionBias: Bool
    public let blockSize: Int
    public let dflashConfig: DFlashInnerConfig

    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size"
        case numHiddenLayers = "num_hidden_layers"
        case numAttentionHeads = "num_attention_heads"
        case numKeyValueHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case intermediateSize = "intermediate_size"
        case rmsNormEps = "rms_norm_eps"
        case ropeTheta = "rope_theta"
        case ropeParameters = "rope_parameters"
        case attentionBias = "attention_bias"
        case blockSize = "block_size"
        case dflashConfig = "dflash_config"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        hiddenSize = try container.decode(Int.self, forKey: .hiddenSize)
        numHiddenLayers = try container.decode(Int.self, forKey: .numHiddenLayers)
        numAttentionHeads = try container.decode(Int.self, forKey: .numAttentionHeads)
        numKeyValueHeads = try container.decodeIfPresent(Int.self, forKey: .numKeyValueHeads)
            ?? numAttentionHeads
        headDim = try container.decodeIfPresent(Int.self, forKey: .headDim)
            ?? hiddenSize / numAttentionHeads
        intermediateSize = try container.decode(Int.self, forKey: .intermediateSize)
        rmsNormEps = try container.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-6
        let rope = try container.decodeIfPresent(DFlashRoPEParameters.self, forKey: .ropeParameters)
        ropeTheta = try container.decodeIfPresent(Float.self, forKey: .ropeTheta)
            ?? rope?.ropeTheta
            ?? 1_000_000
        attentionBias = try container.decodeIfPresent(Bool.self, forKey: .attentionBias) ?? false
        dflashConfig = try container.decode(DFlashInnerConfig.self, forKey: .dflashConfig)
        blockSize = try container.decodeIfPresent(Int.self, forKey: .blockSize)
            ?? dflashConfig.blockSize
            ?? { throw DecodingError.keyNotFound(
                CodingKeys.blockSize,
                .init(codingPath: container.codingPath, debugDescription: "Missing block size")
            ) }()
    }
}

private final class DFlashAttention: Module {
    let config: DFlashDrafterConfiguration
    let scale: Float
    @ModuleInfo(key: "q_proj") var query: Linear
    @ModuleInfo(key: "k_proj") var key: Linear
    @ModuleInfo(key: "v_proj") var value: Linear
    @ModuleInfo(key: "o_proj") var output: Linear
    @ModuleInfo(key: "q_norm") var queryNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var keyNorm: RMSNorm
    let rope: RoPE

    init(_ config: DFlashDrafterConfiguration) {
        self.config = config
        scale = 1 / Float(config.headDim).squareRoot()
        _query.wrappedValue = Linear(
            config.hiddenSize, config.numAttentionHeads * config.headDim, bias: config.attentionBias
        )
        _key.wrappedValue = Linear(
            config.hiddenSize, config.numKeyValueHeads * config.headDim, bias: config.attentionBias
        )
        _value.wrappedValue = Linear(
            config.hiddenSize, config.numKeyValueHeads * config.headDim, bias: config.attentionBias
        )
        _output.wrappedValue = Linear(
            config.numAttentionHeads * config.headDim, config.hiddenSize, bias: config.attentionBias
        )
        _queryNorm.wrappedValue = RMSNorm(dimensions: config.headDim, eps: config.rmsNormEps)
        _keyNorm.wrappedValue = RMSNorm(dimensions: config.headDim, eps: config.rmsNormEps)
        rope = RoPE(dimensions: config.headDim, traditional: false, base: config.ropeTheta, scale: 1)
    }

    func callAsFunction(
        hidden: MLXArray,
        context: MLXArray,
        positionIDs: MLXArray
    ) -> MLXArray {
        let batch = hidden.dim(0)
        let length = hidden.dim(1)
        let contextLength = context.dim(1)
        var queries = query(hidden).reshaped(
            batch, length, config.numAttentionHeads, config.headDim
        )
        queries = queryNorm(queries).transposed(0, 2, 1, 3)
        var keys = concatenated([key(context), key(hidden)], axis: 1).reshaped(
            batch, contextLength + length, config.numKeyValueHeads, config.headDim
        )
        let values = concatenated([value(context), value(hidden)], axis: 1).reshaped(
            batch, contextLength + length, config.numKeyValueHeads, config.headDim
        )
        .transposed(0, 2, 1, 3)
        keys = keyNorm(keys).transposed(0, 2, 1, 3)
        queries = rope(queries, offset: positionIDs.asArray(Int.self).first ?? 0)
        keys = rope(keys, offset: 0)
        return output(MLXFast.scaledDotProductAttention(
            queries: queries,
            keys: keys,
            values: values,
            scale: scale,
            mask: .none
        ).transposed(0, 2, 1, 3).reshaped(batch, length, -1))
    }
}

private final class DFlashMLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gate: Linear
    @ModuleInfo(key: "down_proj") var down: Linear
    @ModuleInfo(key: "up_proj") var upProjection: Linear

    init(_ config: DFlashDrafterConfiguration) {
        _gate.wrappedValue = Linear(config.hiddenSize, config.intermediateSize, bias: false)
        _down.wrappedValue = Linear(config.intermediateSize, config.hiddenSize, bias: false)
        _upProjection.wrappedValue = Linear(config.hiddenSize, config.intermediateSize, bias: false)
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        down(silu(gate(input)) * upProjection(input))
    }
}

private final class DFlashDecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var attention: DFlashAttention
    @ModuleInfo(key: "input_layernorm") var inputNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionNorm: RMSNorm
    let mlp: DFlashMLP

    init(_ config: DFlashDrafterConfiguration) {
        _attention.wrappedValue = DFlashAttention(config)
        _inputNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _postAttentionNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        mlp = DFlashMLP(config)
    }

    func callAsFunction(hidden: MLXArray, context: MLXArray, positionIDs: MLXArray) -> MLXArray {
        let attentionOutput = hidden + attention(
            hidden: inputNorm(hidden), context: context, positionIDs: positionIDs
        )
        return attentionOutput + mlp(postAttentionNorm(attentionOutput))
    }
}

public final class DFlashDraftModel: Module, @unchecked Sendable {
    public let config: DFlashDrafterConfiguration
    private let layers: [DFlashDecoderLayer]
    private let norm: RMSNorm
    @ModuleInfo(key: "fc") var contextProjection: Linear
    @ModuleInfo(key: "hidden_norm") var contextNorm: RMSNorm

    public init(_ config: DFlashDrafterConfiguration) {
        self.config = config
        layers = (0 ..< config.numHiddenLayers).map { _ in DFlashDecoderLayer(config) }
        norm = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _contextProjection.wrappedValue = Linear(
            config.dflashConfig.targetLayerIDs.count * config.hiddenSize,
            config.hiddenSize,
            bias: false
        )
        _contextNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
    }

    public func callAsFunction(
        noiseEmbedding: MLXArray,
        targetHidden: MLXArray,
        positionIDs: MLXArray
    ) -> MLXArray {
        let context = contextNorm(contextProjection(targetHidden))
        var hidden = noiseEmbedding
        for layer in layers {
            hidden = layer(hidden: hidden, context: context, positionIDs: positionIDs)
        }
        return norm(hidden)
    }
}
