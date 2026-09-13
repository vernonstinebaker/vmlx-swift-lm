// DFlash 2 block-diffusion speculative runtime — LLMServerPlus line.
//
// Algorithm reference: osaurus-ai/vmlx-swift SpecDec/DFlash2TokenIterator.swift
// (itself a port of z-lab/dflash `model_mlx.py`). Re-expressed in this line's
// SpecDec runtime shape (synchronous whole-generation run, fresh cache per
// call — the same reprefill contract SpecDecRuntimeLinear uses). Divergences
// from the monorepo iterator, all recorded in
// docs/dflash2-design-notes.md: no CacheCoordinator interplay, no disk-tier
// restore, no compiled/staged verify, no adaptive block-size controller, no
// verify prefetch pipelining, greedy-only (the strategy iterator refuses
// sampling before this runtime is reached).

import Foundation
import MLX

/// Ported from the monorepo iterator (DFlash2TokenIterator.swift); kept
/// name-compatible so both lines read the same.
public enum DFlash2RuntimeError: Error, LocalizedError {
    case emptyPrompt
    case maxTokensTooSmall
    case unsupportedSampling(String)
    case targetLacksPrefixCommitRecording
    case drafterTargetMismatch(String)
    case blockSizeTooSmall(Int)

    public var errorDescription: String? {
        switch self {
        case .emptyPrompt:
            "DFlash 2 requires a non-empty prompt"
        case .maxTokensTooSmall:
            "DFlash 2 requires maxTokens > 1; use the plain iterator for one-token probes"
        case .unsupportedSampling(let detail):
            "DFlash 2 cannot serve this request: \(detail)"
        case .targetLacksPrefixCommitRecording:
            "DFlash 2 needs recurrent verify-input stashing to roll back a hybrid target, and this model does not provide it"
        case .drafterTargetMismatch(let detail):
            "DFlash 2 drafter does not match this model: \(detail)"
        case .blockSizeTooSmall(let value):
            "DFlash 2 block size must be at least 2, got \(value)"
        }
    }
}

public struct DFlash2RuntimeArgs: @unchecked Sendable {
    public let target: any (HiddenStateCaptureModel & TokenEmbedderModel & DFlash2VerifyRollbackModel)
    public let drafter: DFlash2DraftModel
    public let targetLayerIDs: [Int]
    public let maskTokenID: Int32
    public let inputIds: MLXArray
    public let maxNewTokens: Int
    public let stopTokenIDs: Set<Int32>

    public init(
        target: any (HiddenStateCaptureModel & TokenEmbedderModel & DFlash2VerifyRollbackModel),
        drafter: DFlash2DraftModel,
        targetLayerIDs: [Int],
        maskTokenID: Int32,
        inputIds: MLXArray,
        maxNewTokens: Int,
        stopTokenIDs: Set<Int32> = []
    ) {
        self.target = target
        self.drafter = drafter
        self.targetLayerIDs = targetLayerIDs
        self.maskTokenID = maskTokenID
        self.inputIds = inputIds
        self.maxNewTokens = maxNewTokens
        self.stopTokenIDs = stopTokenIDs
    }
}

public struct DFlash2RuntimeResult: Sendable {
    public let tokenIds: [Int32]
}

public enum SpecDecRuntimeDFlash2 {
    /// Retained trailing prompt hidden states for the first propose. The
    /// drafter's own sliding window bounds what its attention can see, so
    /// anything older is discarded by it anyway.
    static let prefillHiddenLimit = 2048

    public static func run(_ args: DFlash2RuntimeArgs) throws -> DFlash2RuntimeResult {
        let blockSize = args.drafter.config.blockSize
        guard blockSize >= 2 else { throw DFlash2RuntimeError.blockSizeTooSmall(blockSize) }
        guard args.maxNewTokens > 1 else { throw DFlash2RuntimeError.maxTokensTooSmall }
        precondition(args.inputIds.ndim == 2 && args.inputIds.dim(0) == 1)

        let captureLayerIDs = Set(args.targetLayerIDs)
        let orderedLayerIDs = args.targetLayerIDs

        var cache = try args.target.newCache(parameters: nil)
        var draftCache = args.drafter.makeCache()

        // Prefill with hidden-state capture; keep only the trailing rows the
        // drafter can condition on.
        let (prefillLogits, captured) = args.target.callAsFunction(
            args.inputIds, cache: cache, captureLayerIDs: captureLayerIDs,
            recordPrefixCommitStates: false)
        var contextHidden = extractContextFeature(
            captured: captured, targetLayerIDs: orderedLayerIDs
        )
        let promptRows = args.inputIds.dim(1)
        if contextHidden.dim(1) > prefillHiddenLimit {
            contextHidden = contextHidden[0..., (contextHidden.dim(1) - prefillHiddenLimit)..., 0...]
        }
        let lastLogits = prefillLogits[0..., (prefillLogits.dim(1) - 1)..., 0...]
        var lastToken = argMax(lastLogits, axis: -1).item(Int32.self)
        eval(lastToken, contextHidden)

        var tokenIds: [Int32] = [lastToken]
        var emitted = 1

        while emitted < args.maxNewTokens {
            let remaining = args.maxNewTokens - emitted
            let bs = Swift.min(blockSize, remaining + 1)
            if bs < 2 {
                // Tail budget cannot fill a block: one plain step.
                let input = MLXArray([lastToken]).reshaped(1, 1)
                let (logits, _) = args.target.callAsFunction(
                    input, cache: cache, captureLayerIDs: captureLayerIDs,
                    recordPrefixCommitStates: false)
                let sampled = argMax(logits[0..., -1, 0...], axis: -1).item(Int32.self)
                eval(sampled)
                tokenIds.append(sampled)
                emitted += 1
                if args.stopTokenIDs.contains(sampled) { break }
                lastToken = sampled
                continue
            }

            // MARK: draft — one forward proposes the whole block.
            var blockIds = [lastToken]
            blockIds.append(contentsOf: repeatElement(args.maskTokenID, count: bs - 1))
            let block = MLXArray(blockIds).reshaped(1, bs)

            // A degraded drafter forward (the monorepo's "husk") surfaces as
            // a wrong-rank proposal here; the guard falls back to AR so a
            // drafter failure costs acceptance, never the turn.
            let proposal = args.drafter.propose(
                inputs: block,
                targetHidden: contextHidden,
                cache: draftCache,
                embedder: args.target,
                temperature: 0,
                logitsStart: 1)
            guard proposal.tokens.ndim == 2, proposal.tokens.dim(1) == bs - 1 else {
                return autoregressiveTail(
                    args, cache: cache, lastToken: lastToken, emitted: emitted,
                    accumulated: tokenIds)
            }
            asyncEval(proposal.tokens)

            // The drafter's sliding clip can advance its cache offset past the
            // committed token count; pull it back so RoPE offsets stay absolute.
            let expectedDraftOffset = promptRows + emitted - 1
            if let head = draftCache.first, head.offset > expectedDraftOffset {
                let excess = head.offset - expectedDraftOffset
                for c in draftCache {
                    if c.isTrimmable {
                        _ = c.trim(excess)
                    } else {
                        c.offsetForDFlash2 = Swift.max(0, c.offset - excess)
                    }
                }
            }

            // MARK: verify — one target forward over [anchor] + drafts. Built
            // on-graph from the drafter output; the single host sync of the
            // cycle happens at the acceptance read below.
            let verifyInput = concatenated(
                [MLXArray([lastToken]).reshaped(1, 1), proposal.tokens], axis: 1)
            // Qwen3.8-class hybrids carry non-trimmable recurrent layers;
            // their MambaCache stashes verify inputs under the policy below.
            let hasRecurrentState = cache.contains { !$0.isTrimmable }
            // The target conforms to DFlash2VerifyRollbackModel (enforced by
            // the args type), so the eager input_capture rollback is always
            // available; capture_commit is intentionally not offered.
            let (logits, captured) = NativeMTPVerifierStatePolicy.withVerifierMode("input_capture") {
                args.target.callAsFunction(
                    verifyInput, cache: cache, captureLayerIDs: captureLayerIDs,
                    recordPrefixCommitStates: false)
            }
            let greedyTargetIds = argMax(logits, axis: -1)
            let newHidden = extractContextFeature(
                captured: captured, targetLayerIDs: orderedLayerIDs)
            asyncEval(greedyTargetIds, newHidden)

            // MARK: accept
            let draftIDs = proposal.tokens.reshaped(-1).asArray(Int32.self)
            let targetIDs = greedyTargetIds.reshaped(-1).asArray(Int32.self)
            let acceptance = DFlash2Sampling.acceptGreedy(
                draftTokens: draftIDs.map(Int.init), targetTokens: targetIDs.map(Int.init))
            let accepted = acceptance.accepted
            let committedInputs = accepted + 1
            let rejected = verifyInput.dim(1) - committedInputs

            // MARK: commit
            if rejected > 0 {
                for layer in cache where layer.isTrimmable {
                    _ = layer.trim(rejected)
                }
                guard args.target.commitVerifiedBlock(
                    cache: cache, acceptedInputs: committedInputs)
                else {
                    // The recurrent layers had no stash for the accepted
                    // prefix; the cache can no longer be trusted. Truncating
                    // loudly beats emitting tokens the target never produced.
                    throw DFlash2RuntimeError.targetLacksPrefixCommitRecording
                }
            } else {
                for layer in cache { (layer as? MambaCache)?.clearVerifyInputStash() }
            }
            contextHidden = newHidden[0..., ..<committedInputs, 0...]

            var newTokens = Array(draftIDs.prefix(accepted))
            let bonus = Int32(acceptance.bonus)
            newTokens.append(bonus)
            var hitStop = false
            for token in newTokens {
                tokenIds.append(token)
                emitted += 1
                if args.stopTokenIDs.contains(token) {
                    hitStop = true
                    break
                }
            }
            if hitStop { break }
            lastToken = newTokens.last ?? lastToken
        }

        return DFlash2RuntimeResult(tokenIds: tokenIds)
    }

    /// Finish the remaining budget with plain single-token decode — the
    /// safety valve when the drafter cannot propose.
    private static func autoregressiveTail(
        _ args: DFlash2RuntimeArgs,
        cache: [KVCache],
        lastToken: Int32,
        emitted: Int,
        accumulated: [Int32]
    ) -> DFlash2RuntimeResult {
        var tokenIds = accumulated
        var last = lastToken
        var count = emitted
        while count < args.maxNewTokens {
            let input = MLXArray([last]).reshaped(1, 1)
            let (logits, _) = args.target.callAsFunction(
                input, cache: cache, captureLayerIDs: [],
                recordPrefixCommitStates: false)
            let sampled = argMax(logits[0..., -1, 0...], axis: -1).item(Int32.self)
            eval(sampled)
            tokenIds.append(sampled)
            count += 1
            if args.stopTokenIDs.contains(sampled) { break }
            last = sampled
        }
        return DFlash2RuntimeResult(tokenIds: tokenIds)
    }
}
