import Foundation

class SenseVoiceASREngine: ASREngineProtocol {
    var supportsStreaming: Bool { false }

    private var context: SenseVoiceContext?

    func initialize(modelPath: String) -> Bool {
        context = SenseVoiceContext(modelPath: modelPath)
        return context != nil
    }

    func transcribe(_ pcmData: Data) -> ASRResult {
        guard let context = context,
              let bridgeResult = context.transcribeData(withMetrics: pcmData) else {
            return ASRResult(text: "Error: Engine not initialized.", metrics: nil)
        }

        let metrics = ASRSessionMetrics(from: bridgeResult)
        let text = bridgeResult.text ?? "No text detected."
        return ASRResult(text: text, metrics: metrics)
    }
}
