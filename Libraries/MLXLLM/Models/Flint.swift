// Copyright 2026 RecursiveSI / NovaMLX port. SPDX-License-Identifier: Apache-2.0
//
// Port of mlx_lm/models/flint.py, adapted to the upstream MLXLMCommon cache
// and attention APIs. Flint is a 49-layer hybrid GQA/MoE decoder with learned
// per-head attention sinks.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

public struct FlintConfiguration: Decodable, Sendable {
    public let modelType: String
    public let hiddenSize: Int
    public let hiddenLayers: Int
    public let intermediateSize: Int
    public let attentionHeads: Int
    public let kvHeads: Int
    public let headDim: Int
    public let numExperts: Int
    public let numExpertsPerTok: Int
    public let moeIntermediateSize: Int
    public let sharedExpertIntermediateSize: Int
    public let mlpOnlyLayers: [Int]
    public let layerTypes: [String]
    public let slidingWindow: Int
    public let rmsNormEps: Float
    public let vocabularySize: Int
    public let ropeTheta: Float
    public let maxPositionEmbeddings: Int?
    public let tieWordEmbeddings: Bool

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case intermediateSize = "intermediate_size"
        case attentionHeads = "num_attention_heads"
        case kvHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case numExperts = "num_experts"
        case numExpertsPerTok = "num_experts_per_tok"
        case moeIntermediateSize = "moe_intermediate_size"
        case sharedExpertIntermediateSize = "shared_expert_intermediate_size"
        case mlpOnlyLayers = "mlp_only_layers"
        case layerTypes = "layer_types"
        case slidingWindow = "sliding_window"
        case rmsNormEps = "rms_norm_eps"
        case vocabularySize = "vocab_size"
        case ropeTheta = "rope_theta"
        case ropeParameters = "rope_parameters"
        case maxPositionEmbeddings = "max_position_embeddings"
        case tieWordEmbeddings = "tie_word_embeddings"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        modelType = try c.decode(String.self, forKey: .modelType)
        hiddenSize = try c.decode(Int.self, forKey: .hiddenSize)
        hiddenLayers = try c.decode(Int.self, forKey: .hiddenLayers)
        intermediateSize = try c.decode(Int.self, forKey: .intermediateSize)
        attentionHeads = try c.decode(Int.self, forKey: .attentionHeads)
        kvHeads = try c.decode(Int.self, forKey: .kvHeads)
        headDim = try c.decode(Int.self, forKey: .headDim)
        numExperts = try c.decode(Int.self, forKey: .numExperts)
        numExpertsPerTok = try c.decode(Int.self, forKey: .numExpertsPerTok)
        moeIntermediateSize = try c.decode(Int.self, forKey: .moeIntermediateSize)
        sharedExpertIntermediateSize = try c.decode(Int.self, forKey: .sharedExpertIntermediateSize)
        mlpOnlyLayers = try c.decodeIfPresent([Int].self, forKey: .mlpOnlyLayers) ?? []
        layerTypes = try c.decodeIfPresent([String].self, forKey: .layerTypes) ?? []
        slidingWindow = try c.decode(Int.self, forKey: .slidingWindow)
        rmsNormEps = try c.decode(Float.self, forKey: .rmsNormEps)
        vocabularySize = try c.decode(Int.self, forKey: .vocabularySize)
        let nestedRope = try c.decodeIfPresent(
            FlintRopeParameters.self, forKey: .ropeParameters)
        let flatRope = try c.decodeIfPresent(Float.self, forKey: .ropeTheta)
        ropeTheta = nestedRope?.ropeTheta.map(Float.init) ?? flatRope ?? 10000
        maxPositionEmbeddings = try c.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings)
        tieWordEmbeddings = try c.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? false
    }
}

private struct FlintRopeParameters: Decodable {
    let ropeTheta: Double?

    enum CodingKeys: String, CodingKey {
        case ropeTheta = "rope_theta"
    }
}

private func attentionWithCacheUpdateAndSinks(
    queries: MLXArray,
    keys: MLXArray,
    values: MLXArray,
    cache: KVCache?,
    scale: Float,
    mask: MLXFast.ScaledDotProductAttentionMaskMode = .none,
    sinks: MLXArray? = nil
) -> MLXArray {
    guard let cache else {
        return MLXFast.scaledDotProductAttention(
            queries: queries, keys: keys, values: values,
            scale: scale, mask: mask, sinks: sinks)
    }

    if let attentionCache = cache as? KVCacheAttentionProtocol, sinks == nil {
        return attentionCache.updateAndAttend(
            queries: queries, keys: keys, values: values, scale: scale, mask: mask)
    } else if let quantizedKVCache = cache as? QuantizedKVCacheProtocol {
        precondition(sinks == nil, "Quantized SDPA does not support attention sinks.")
        let (quantizedKeys, quantizedValues) = quantizedKVCache.updateQuantized(
            keys: keys, values: values)
        return quantizedScaledDotProductAttention(
            queries: queries, quantizedKeys: quantizedKeys, quantizedValues: quantizedValues,
            scale: scale, mask: mask, groupSize: quantizedKVCache.groupSize,
            bits: quantizedKVCache.bits, mode: quantizedKVCache.mode)
    } else {
        let (cachedKeys, cachedValues) = cache.update(keys: keys, values: values)
        return MLXFast.scaledDotProductAttention(
            queries: queries, keys: cachedKeys, values: cachedValues,
            scale: scale, mask: mask, sinks: sinks)
    }
}

private final class FlintAttention: Module {
    let args: FlintConfiguration
    let headDim: Int
    let scale: Float

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear
    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm
    @ParameterInfo(key: "softmax_offset") var softmaxOffset: MLXArray

    let rope: RoPELayer?

    init(_ args: FlintConfiguration, layerIdx: Int) {
        self.args = args
        self.headDim = args.headDim
        self.scale = pow(Float(args.headDim), -0.5)
        let dim = args.hiddenSize

        _qProj.wrappedValue = Linear(dim, args.attentionHeads * args.headDim, bias: false)
        _kProj.wrappedValue = Linear(dim, args.kvHeads * args.headDim, bias: false)
        _vProj.wrappedValue = Linear(dim, args.kvHeads * args.headDim, bias: false)
        _oProj.wrappedValue = Linear(args.attentionHeads * args.headDim, dim, bias: false)
        _qNorm.wrappedValue = RMSNorm(dimensions: args.headDim, eps: args.rmsNormEps)
        _kNorm.wrappedValue = RMSNorm(dimensions: args.headDim, eps: args.rmsNormEps)
        _softmaxOffset.wrappedValue = zeros([args.attentionHeads])

        if !args.layerTypes.isEmpty, args.layerTypes[layerIdx] == "sliding_attention" {
            self.rope = initializeRope(
                dims: args.headDim,
                base: args.ropeTheta,
                traditional: false,
                scalingConfig: nil,
                maxPositionEmbeddings: args.maxPositionEmbeddings)
        } else {
            self.rope = nil
        }
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let (b, l) = (x.dim(0), x.dim(1))
        var q = qNorm(qProj(x).reshaped(b, l, args.attentionHeads, -1))
            .transposed(0, 2, 1, 3)
        var k = kNorm(kProj(x).reshaped(b, l, args.kvHeads, -1))
            .transposed(0, 2, 1, 3)
        let v = vProj(x).reshaped(b, l, args.kvHeads, -1).transposed(0, 2, 1, 3)

        if let rope {
            let offset = cache?.ropeOffset
            q = applyRotaryPosition(rope, to: q, offset: offset)
            k = applyRotaryPosition(rope, to: k, offset: offset)
        }

        let out = attentionWithCacheUpdateAndSinks(
            queries: q, keys: k, values: v, cache: cache,
            scale: scale, mask: mask, sinks: softmaxOffset)
        return oProj(out.transposed(0, 2, 1, 3).reshaped(b, l, -1))
    }
}

private final class FlintMLP: Module {
    @ModuleInfo(key: "gate_proj") var gate: Linear
    @ModuleInfo(key: "down_proj") var down: Linear
    @ModuleInfo(key: "up_proj") var up: Linear

    init(dimensions: Int, hiddenDimensions: Int) {
        _gate.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
        _down.wrappedValue = Linear(hiddenDimensions, dimensions, bias: false)
        _up.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        down(silu(gate(x)) * up(x))
    }
}

private final class FlintSparseMoeBlock: Module {
    let topK: Int

    @ModuleInfo(key: "gate") var gate: Linear
    @ModuleInfo(key: "switch_mlp") var switchMLP: SwitchGLU
    @ModuleInfo(key: "shared_expert") var sharedExpert: FlintMLP

    init(_ args: FlintConfiguration) {
        self.topK = args.numExpertsPerTok
        _gate.wrappedValue = Linear(args.hiddenSize, args.numExperts, bias: false)
        _switchMLP.wrappedValue = SwitchGLU(
            inputDims: args.hiddenSize,
            hiddenDims: args.moeIntermediateSize,
            numExperts: args.numExperts)
        _sharedExpert.wrappedValue = FlintMLP(
            dimensions: args.hiddenSize, hiddenDimensions: args.sharedExpertIntermediateSize)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let logits = gate(x.asType(.float32))
        let scores = sigmoid(logits)
        let indices = argPartition(-scores, kth: topK - 1, axis: -1)[.ellipsis, ..<topK]
        var selected = takeAlong(scores, indices, axis: -1)
        selected = selected / selected.sum(axis: -1, keepDims: true)

        let y = switchMLP(x, indices)
        let combined = (y * selected.asType(y.dtype)[.ellipsis, .newAxis]).sum(axis: -2)
        return combined + sharedExpert(x)
    }
}

private final class FlintDecoderLayer: Module {
    let useSliding: Bool

    @ModuleInfo(key: "self_attn") var selfAttn: FlintAttention
    @ModuleInfo(key: "mlp") var mlp: Module
    @ModuleInfo(key: "input_layernorm") var inputNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postNorm: RMSNorm

    init(_ args: FlintConfiguration, layerIdx: Int) {
        self.useSliding =
            !args.layerTypes.isEmpty && args.layerTypes[layerIdx] == "sliding_attention"
        _selfAttn.wrappedValue = FlintAttention(args, layerIdx: layerIdx)
        if args.mlpOnlyLayers.contains(layerIdx) {
            _mlp.wrappedValue = FlintMLP(
                dimensions: args.hiddenSize, hiddenDimensions: args.intermediateSize)
        } else {
            _mlp.wrappedValue = FlintSparseMoeBlock(args)
        }
        _inputNorm.wrappedValue = RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)
        _postNorm.wrappedValue = RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let h = x + selfAttn(inputNorm(x), mask: mask, cache: cache)
        let normed = postNorm(h)
        if let moe = mlp as? FlintSparseMoeBlock {
            return h + moe(normed)
        }
        return h + (mlp as! FlintMLP)(normed)
    }
}

private final class FlintModelInner: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo(key: "norm") var norm: RMSNorm
    let layerTypes: [String]
    let layers: [FlintDecoderLayer]
    let windowSize: Int
    let slidingAttentionIndex: Int
    let fullAttentionIndex: Int
    let config: FlintConfiguration

    init(_ config: FlintConfiguration) {
        self.config = config
        _embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabularySize, dimensions: config.hiddenSize)
        _norm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self.layerTypes = config.layerTypes
        self.layers = (0 ..< config.hiddenLayers).map { FlintDecoderLayer(config, layerIdx: $0) }
        self.windowSize = config.slidingWindow
        self.slidingAttentionIndex = layerTypes.firstIndex(of: "sliding_attention") ?? 0
        self.fullAttentionIndex = layerTypes.firstIndex(of: "full_attention") ?? 0
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        var x = embedTokens(inputs)
        let caches: [KVCache?] = cache ?? [KVCache?](repeating: nil, count: layers.count)
        let seqLen = x.dim(1)
        var fullMask: MLXFast.ScaledDotProductAttentionMaskMode?
        var slidingMask: MLXFast.ScaledDotProductAttentionMaskMode?

        for (i, layer) in layers.enumerated() {
            let maskMode: MLXFast.ScaledDotProductAttentionMaskMode
            if layerTypes[i] == "full_attention" {
                if fullMask == nil {
                    fullMask = makeAttentionMask(
                        n: seqLen, cache: caches[fullAttentionIndex], windowSize: nil)
                }
                maskMode = fullMask!
            } else {
                if slidingMask == nil {
                    slidingMask = makeAttentionMask(
                        n: seqLen, cache: caches[slidingAttentionIndex], windowSize: windowSize)
                }
                maskMode = slidingMask!
            }
            x = layer(x, mask: maskMode, cache: caches[i])
        }
        return norm(x)
    }
}

public final class FlintModel: Module, LLMModel, KVCacheDimensionProvider {
    public let vocabularySize: Int
    public let kvHeads: [Int]

    private let model: FlintModelInner
    public let configuration: FlintConfiguration

    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    public var loraLayers: [Module] { model.layers }
    public var toolCallFormat: ToolCallFormat? { .glm4 }

    public init(_ args: FlintConfiguration) {
        self.configuration = args
        self.vocabularySize = args.vocabularySize
        self.kvHeads = Array(repeating: args.kvHeads, count: args.hiddenLayers)
        self.model = FlintModelInner(args)
        if !args.tieWordEmbeddings {
            _lmHead.wrappedValue = Linear(args.hiddenSize, args.vocabularySize, bias: false)
        }
        super.init()
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        let out = model(inputs, cache: cache)
        if let lmHead {
            return lmHead(out)
        }
        return model.embedTokens.asLinear(out)
    }

    public func callAsFunction(
        _ input: LMInput.Text, cache: [KVCache]?, state: LMOutput.State?
    ) -> LMOutput {
        let hidden = model(input.tokens, cache: cache)
        let last = hidden[0..., (hidden.dim(1) - 1) ..< hidden.dim(1), 0...]
        if let lmHead {
            return LMOutput(logits: lmHead(last))
        }
        return LMOutput(logits: model.embedTokens.asLinear(last))
    }

    public func newCache(parameters: GenerateParameters?) throws -> [KVCache] {
        model.layerTypes.map { layerType in
            layerType == "sliding_attention"
                ? RotatingKVCache(maxSize: configuration.slidingWindow, keep: 0)
                : KVCacheSimple()
        }
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var result = weights

        // RecursiveSI/flint-int4 stores packed experts as uint8; view them as
        // uint32 so the loader's quantization pass pairs them with their scales.
        var packed: [String] = []
        for (key, value) in result where value.dtype == .uint8 {
            let base = key.hasSuffix(".weight") ? String(key.dropLast(7)) : key
            if result["\(base).scales"] != nil {
                packed.append(key)
            }
        }
        for key in packed {
            result[key] = result[key]!.view(dtype: .uint32)
        }

        // Tolerate per-expert sharded bundles by stacking them when the full
        // set is present. The published flint-int4 bundle already ships
        // switch_mlp, so this is a no-op for it.
        for layer in 0 ..< configuration.hiddenLayers
        where !configuration.mlpOnlyLayers.contains(layer) {
            let prefix = "model.layers.\(layer).mlp"
            for name in ["up_proj", "down_proj", "gate_proj"] {
                guard result["\(prefix).experts.0.\(name).weight"] != nil,
                    result["\(prefix).experts.\(configuration.numExperts - 1).\(name).weight"]
                        != nil
                else { continue }
                let experts = (0 ..< configuration.numExperts).map { expert in
                    result.removeValue(forKey: "\(prefix).experts.\(expert).\(name).weight")!
                }
                result["\(prefix).switch_mlp.\(name).weight"] = stacked(experts)
            }
        }

        return result
    }
}
