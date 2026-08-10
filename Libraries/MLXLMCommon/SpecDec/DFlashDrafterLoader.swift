import Foundation
import MLX
import MLXNN

public enum DFlashDrafterLoader {
    public static func load(from directory: URL) throws -> DFlashDraftModel {
        let directory = directory.resolvingSymlinksInPath()
        let configURL = directory.appending(component: "config.json")
        guard FileManager.default.fileExists(atPath: configURL.path) else {
            throw DFlashDrafterLoadError.missingConfig(configURL)
        }
        let configuration: DFlashDrafterConfiguration
        do {
            configuration = try JSONDecoder().decode(
                DFlashDrafterConfiguration.self,
                from: Data(contentsOf: configURL)
            )
        } catch {
            throw DFlashDrafterLoadError.malformedConfig(configURL, error)
        }
        var weights: [String: MLXArray] = [:]
        let enumerator = FileManager.default.enumerator(at: directory, includingPropertiesForKeys: nil)
        if let enumerator {
            for case let url as URL in enumerator where url.pathExtension == "safetensors" {
                let (loaded, _) = try loadArraysAndMetadata(url: url)
                weights.merge(loaded) { _, new in new }
            }
        }
        guard !weights.isEmpty else { throw DFlashDrafterLoadError.noWeights(directory) }
        let model = DFlashDraftModel(configuration)
        do {
            try model.update(parameters: ModuleParameters.unflattened(weights), verify: [.noUnusedKeys])
        } catch {
            throw DFlashDrafterLoadError.weightUpdateFailed(error)
        }
        MLX.eval(model)
        return model
    }

    public static func looksLikeDrafter(at directory: URL) -> Bool {
        let config = directory.appending(component: "config.json")
        guard let data = try? Data(contentsOf: config),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        else { return false }
        return json["dflash_config"] != nil
    }
}

public enum DFlashDrafterLoadError: Error, LocalizedError {
    case missingConfig(URL)
    case malformedConfig(URL, Error)
    case noWeights(URL)
    case weightUpdateFailed(Error)

    public var errorDescription: String? {
        switch self {
        case let .missingConfig(url): "DFlash drafter is missing \(url.path)"
        case let .malformedConfig(url, error): "DFlash drafter config at \(url.path) is invalid: \(error)"
        case let .noWeights(url): "DFlash drafter has no safetensors weights at \(url.path)"
        case let .weightUpdateFailed(error): "DFlash drafter weights could not be loaded: \(error)"
        }
    }
}
