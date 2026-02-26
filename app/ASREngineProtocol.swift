import Foundation

enum ASRModel: String, CaseIterable, Identifiable {
    case senseVoice = "SenseVoice"
    case paraformer = "Paraformer"

    var id: String { rawValue }

    var supportsStreaming: Bool {
        switch self {
        case .paraformer: return true
        case .senseVoice: return false
        }
    }
}

struct ASRResult {
    let text: String
    let metrics: ASRSessionMetrics?
}

protocol ASREngineProtocol: AnyObject {
    var supportsStreaming: Bool { get }
    func initialize(modelPath: String) -> Bool
    func transcribe(_ pcmData: Data) -> ASRResult

    // Streaming interface
    func startStream()
    func feedChunk(_ pcmData: Data) -> String?
    func endStream() -> ASRResult
}

// Default no-op streaming implementations for non-streaming engines
extension ASREngineProtocol {
    func startStream() {}
    func feedChunk(_ pcmData: Data) -> String? { return nil }
    func endStream() -> ASRResult {
        return ASRResult(text: "", metrics: nil)
    }
}
