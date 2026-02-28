import SwiftUI

struct MetricsOverlayView: View {
    let metrics: ASRSessionMetrics

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack(spacing: 12) {
                metricItem("RTF", String(format: "%.3f", metrics.rtf))
                metricItem("Total", String(format: "%.0fms", metrics.totalInferenceMs))
                metricItem("ONNX", String(format: "%.0fms", metrics.onnxForwardTimeMs))
                metricItem("Mem", String(format: "%.1fMB", metrics.memoryMB))
            }
            HStack(spacing: 12) {
                metricItem("Fbank", String(format: "%.0fms", metrics.fbankTimeMs))
                metricItem("LFR+CMVN", String(format: "%.0fms", metrics.lfrCmvnTimeMs))
                metricItem("Decode", String(format: "%.0fms", metrics.decodeTimeMs))
                metricItem("Audio", String(format: "%.0fms", metrics.audioDurationMs))
            }
            HStack(spacing: 12) {
                metricItem("Thermal", metrics.thermalStateDescription)
            }
        }
        .font(.system(size: 11, design: .monospaced))
        .padding(8)
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 8))
    }

    private func metricItem(_ label: String, _ value: String) -> some View {
        HStack(spacing: 2) {
            Text(label)
                .foregroundColor(.secondary)
            Text(value)
                .foregroundColor(.primary)
        }
    }
}
