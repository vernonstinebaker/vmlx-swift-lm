import MLX

enum BatchCompile {
    static func compileForward(
        model: any LanguageModel,
        cache: [CompilableKVCache]
    ) -> @Sendable ([MLXArray]) -> [MLXArray] {
        precondition(!cache.isEmpty)

        let capturedModel = model
        let capturedCache = cache
        let state = capturedCache.map { $0 as any Updatable }
        return compile(inputs: state, outputs: state) { arguments in
            [capturedModel(
                LMInput.Text(tokens: arguments[0])[text: .newAxis],
                cache: capturedCache.map { $0 as KVCache },
                state: nil
            ).logits]
        }
    }
}
