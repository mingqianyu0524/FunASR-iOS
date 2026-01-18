import SwiftUI

struct ContentView: View {
    // 1. 状态管理
    @StateObject private var recorder = AudioRecorder()
    @State private var transcription: String = "Press and hold to record..."
    @State private var isProcessing: Bool = false
    @State private var isModelLoading: Bool = true
    
    // 保持 WhisperContext 的引用
    @State private var whisperContext: WhisperContext?
    
    var body: some View {
        VStack(spacing: 20) {
            // 标题
            Text("Whisper Native")
                .font(.largeTitle)
                .fontWeight(.bold)
                .padding(.top)
            
            // 结果显示区域
            if isModelLoading {
                VStack {
                    ProgressView()
                        .scaleEffect(1.5)
                    Text("Loading Model (Small)...")
                        .font(.caption)
                        .foregroundColor(.gray)
                        .padding(.top, 8)
                }
                .frame(maxHeight: .infinity)
            } else {
                ScrollView {
                    Text(transcription)
                        .font(.body)
                        .padding()
                    .frame(maxWidth: .infinity, alignment: .leading)
                }
                .frame(maxHeight: .infinity)
                .background(Color(UIColor.secondarySystemBackground))
                .cornerRadius(12)
                .padding()
            }
            
            // 声波视图 (Waveform)
            WaveformView(recorder: recorder)
                .frame(height: 120)
                .padding(.horizontal)
            
            // 底部控制区
            Spacer()
            
            // 录音按钮 (按住说话)
            ZStack {
                // 外圈动画 (录音时放大变红)
                Circle()
                    .fill(recorder.isRecording ? Color.red : Color.blue)
                    .frame(width: 80, height: 80)
                    .scaleEffect(recorder.isRecording ? 1.2 : 1.0)
                    .opacity(recorder.isRecording ? 0.5 : 1.0)
                    .animation(.easeInOut(duration: 0.2), value: recorder.isRecording)
                
                // 麦克风图标
                Image(systemName: recorder.isRecording ? "waveform" : "mic.fill")
                    .font(.system(size: 30))
                    .foregroundColor(.white)
            }
            // 使用 DragGesture 模拟 "按住说话" (TouchDown / TouchUp)
            .gesture(
                DragGesture(minimumDistance: 0)
                    .onChanged { _ in
                        if !recorder.isRecording && !isProcessing {
                            startRecording()
                        }
                    }
                    .onEnded { _ in
                        if recorder.isRecording {
                            stopRecordingAndTranscribe()
                        }
                    }
            )
            .disabled(isProcessing || isModelLoading) // 处理时禁用按钮
            .padding(.bottom, 30)
            
            // 加载指示器
            if isProcessing {
                ProgressView("Transcribing...")
                    .padding(.bottom, 10)
            }
        }
        .onAppear {
            // 页面加载时初始化引擎
            initWhisperAsync()
        }
    }
    
    // MARK: - Logic
    
    func initWhisperAsync() {
        print("🚀 Initializing Whisper Engine...")
        DispatchQueue.global(qos: .userInitiated).async {
            if let resourcePath = Bundle.main.resourcePath {
                let context = WhisperContext(modelPath: resourcePath)
                
                // When done, update UI on Main Thread
                DispatchQueue.main.async {
                    self.whisperContext = context
                    self.isModelLoading = false // Hide loading spinner
                    print("✅ Whisper Context Ready (UI Updated).")
                }
            } else {
                DispatchQueue.main.async {
                    self.transcription = "Error: Could not find bundle resource path."
                    self.isModelLoading = false
                }
            }
        }
    }
    
    func startRecording() {
        // 请求权限并开始
        recorder.requestPermission()
        recorder.startRecording()
    }
    
    func stopRecordingAndTranscribe() {
        // 1. 停止录音
        recorder.stopRecording()
        
        guard let context = whisperContext else {
            transcription = "Error: Engine not initialized."
            return
        }
        
        // 简单防抖：如果录音太短，或者没有数据
        if recorder.recordingData.count < 1600 { // < 0.1秒
            print("⚠️ Audio too short.")
            return
        }
        
        isProcessing = true
        let rawData = recorder.recordingData
        
        // 2. 异步执行推理 (避免阻塞 UI 线程)
        DispatchQueue.global(qos: .userInitiated).async {
            print("🧵 Encoding audio data for Bridge...")
            
            // 数据转换: Swift [Float] -> Objective-C NSArray<NSNumber*>
            // (虽然这有一次拷贝开销，但对于 Demo 来说足够快且安全)
            let numberArray = rawData.map { NSNumber(value: $0) }
            
            print("🧠 Running Inference C++...")
            let result = context.transcribe(numberArray)
            
            // 3. 回到主线程更新 UI
            DispatchQueue.main.async {
                self.transcription = result ?? "No text detected."
                self.isProcessing = false
            }
        }
    }
}

// 预览
struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}
