import MLX

enum BatchCompile {
    /// BatchEngine owns the compiled forward's lifetime: it builds the closure
    /// on the engine's execution path, invokes it only under the engine's
    /// decode serialization, and replaces it wholesale on promotion. The model
    /// and cache are engine-owned and outlive every invocation, so the capture
    /// is safe even though neither type is Sendable. (mlx-swift 0.33.0 made
    /// stateless compile bodies @Sendable — #469 — while the Updatable-inputs
    /// overload this uses still returns a plain closure; the boxes bridge the
    /// two.)
    private struct ForwardCaptures: @unchecked Sendable {
        let model: any LanguageModel
        let cache: [CompilableKVCache]
    }

    private struct CompiledForward: @unchecked Sendable {
        let run: ([MLXArray]) -> [MLXArray]
    }

    static func compileForward(
        model: any LanguageModel,
        cache: [CompilableKVCache]
    ) -> @Sendable ([MLXArray]) -> [MLXArray] {
        precondition(!cache.isEmpty)

        let captures = ForwardCaptures(model: model, cache: cache)
        let state = cache.map { $0 as any Updatable }
        let compiled = CompiledForward(
            run: compile(inputs: state, outputs: state) { arguments in
                [captures.model(
                    LMInput.Text(tokens: arguments[0])[text: .newAxis],
                    cache: captures.cache.map { $0 as KVCache },
                    state: nil
                ).logits]
            })
        return { compiled.run($0) }
    }
}
