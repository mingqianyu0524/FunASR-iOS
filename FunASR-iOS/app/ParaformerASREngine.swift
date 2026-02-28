import Foundation

class ParaformerASREngine: ASREngineProtocol {
    var supportsStreaming: Bool { true }

    private var context: ParaformerContext?

    func initialize(modelPath: String) -> Bool {
        context = ParaformerContext(modelPath: modelPath)
        // ParaformerContext returns non-nil even if init fails internally,
        // but the C++ engine will not be initialized. We test by checking
        // if the context was created (ObjC init always returns non-nil here).
        // The actual failure will surface on first transcribe call.
        return context != nil
    }

    func transcribe(_ pcmData: Data) -> ASRResult {
        guard let context = context,
              let bridgeResult = context.transcribeData(withMetrics: pcmData) else {
            return ASRResult(text: "Error: Paraformer engine not initialized.", metrics: nil)
        }

        let metrics = ASRSessionMetrics(from: bridgeResult)
        let text = bridgeResult.text ?? "No text detected."
        return ASRResult(text: text, metrics: metrics)
    }

    func startStream() {
        context?.startStream()
    }

    func feedChunk(_ pcmData: Data) -> String? {
        guard let context = context else { return nil }
        let result = context.feedChunk(pcmData)
        if let r = result, !r.isEmpty {
            return r
        }
        return nil
    }

    func endStream() -> ASRResult {
        guard let context = context,
              let bridgeResult = context.endStreamWithMetrics() else {
            return ASRResult(text: "", metrics: nil)
        }

        let metrics = ASRSessionMetrics(from: bridgeResult)
        let text = bridgeResult.text ?? ""
        return ASRResult(text: text, metrics: metrics)
    }
}
