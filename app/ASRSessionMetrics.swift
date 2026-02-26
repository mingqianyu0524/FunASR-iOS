import Foundation

struct ASRSessionMetrics: Codable {
    let fbankTimeMs: Double
    let lfrCmvnTimeMs: Double
    let onnxForwardTimeMs: Double
    let decodeTimeMs: Double
    let totalInferenceMs: Double
    let audioDurationMs: Double
    let rtf: Double
    let memoryBytesUsed: Int64
    let thermalState: Int  // 0=nominal, 1=fair, 2=serious, 3=critical
    let timestamp: Date

    init(from bridgeResult: ASRBridgeResult) {
        self.fbankTimeMs = bridgeResult.fbankTimeMs
        self.lfrCmvnTimeMs = bridgeResult.lfrCmvnTimeMs
        self.onnxForwardTimeMs = bridgeResult.onnxForwardTimeMs
        self.decodeTimeMs = bridgeResult.decodeTimeMs
        self.totalInferenceMs = bridgeResult.totalInferenceMs
        self.audioDurationMs = bridgeResult.audioDurationMs
        self.rtf = bridgeResult.rtf
        self.memoryBytesUsed = bridgeResult.memoryBytesUsed
        self.thermalState = ProcessInfo.processInfo.thermalState.rawValue
        self.timestamp = Date()
    }

    var memoryMB: Double {
        Double(memoryBytesUsed) / (1024.0 * 1024.0)
    }

    var thermalStateDescription: String {
        switch thermalState {
        case 0: return "Nominal"
        case 1: return "Fair"
        case 2: return "Serious"
        case 3: return "Critical"
        default: return "Unknown"
        }
    }
}
