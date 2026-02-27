import SwiftUI

struct WaveformView: View {
    // 监听录音器的数据变化
    @ObservedObject var recorder: AudioRecorder
    
    // 配置：显示最近多少个采样点 (例如 300 个点)
    private let sampleCountToDisplay = 300
    
    var body: some View {
        GeometryReader { geometry in
            ZStack {
                // 背景
                RoundedRectangle(cornerRadius: 12)
                    .fill(Color.black.opacity(0.8))
                
                // 波形路径
                if recorder.isRecording && !recorder.recordingData.isEmpty {
                    WaveShape(
                        samples: recorder.recordingData,
                        countToDisplay: sampleCountToDisplay
                    )
                    .stroke(Color.green, lineWidth: 2)
                    .frame(width: geometry.size.width, height: geometry.size.height)
                    .padding(8)
                    // 添加一点发光效果让它看起来更酷
                    .shadow(color: .green, radius: 4)
                } else {
                    // 空闲状态显示一条直线
                    Path { path in
                        let midY = geometry.size.height / 2
                        path.move(to: CGPoint(x: 0, y: midY))
                        path.addLine(to: CGPoint(x: geometry.size.width, y: midY))
                    }
                    .stroke(Color.gray.opacity(0.5), lineWidth: 1)
                    .padding(8)
                }
            }
        }
    }
}

// 定义波形形状
struct WaveShape: Shape {
    var samples: [Float]
    var countToDisplay: Int
    
    func path(in rect: CGRect) -> Path {
        var path = Path()
        
        // 获取数据的最后 N 个点
        let suffix = samples.suffix(countToDisplay)
        guard !suffix.isEmpty else { return path }
        
        let width = rect.width
        let height = rect.height
        let midY = height / 2
        
        // X 轴步长: 每个点在屏幕上占多少宽度
        let stepX = width / CGFloat(countToDisplay - 1)
        
        // 转换数据点为屏幕坐标
        // 假设音频数据范围大约在 -1.0 到 1.0 之间
        // 我们将其放大一点以便看得更清楚
        let scale: CGFloat = 0.8
        
        for (index, sample) in suffix.enumerated() {
            let x = CGFloat(index) * stepX
            
            // Y 轴: 翻转并缩放 (sample * scale * height/2)
            // 负 sample 向上，正 sample 向下 (因为屏幕坐标 Y 向下是正)
            let sampleHeight = CGFloat(sample) * (height / 2) * scale
            let y = midY - sampleHeight
            
            if index == 0 {
                path.move(to: CGPoint(x: x, y: y))
            } else {
                path.addLine(to: CGPoint(x: x, y: y))
            }
        }
        
        return path
    }
}

// 预览辅助
struct WaveformView_Previews: PreviewProvider {
    static var previews: some View {
        // 创建一个带有假数据的 Recorder 用于预览
        let mockRecorder = AudioRecorder()
        // 填充一些正弦波数据
        mockRecorder.recordingData = (0..<300).map { Float(sin(Double($0) * 0.1)) * 0.5 }
        mockRecorder.isRecording = true
        
        return WaveformView(recorder: mockRecorder)
            .frame(height: 100)
            .padding()
            .previewLayout(.sizeThatFits)
    }
}
