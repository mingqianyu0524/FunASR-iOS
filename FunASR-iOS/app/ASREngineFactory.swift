import Foundation

enum ASREngineFactory {
    static func create(model: ASRModel) -> any ASREngineProtocol {
        switch model {
        case .senseVoice:
            return SenseVoiceASREngine()
        case .paraformer:
            return ParaformerASREngine()
        }
    }
}
