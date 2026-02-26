import SwiftUI

struct ContentView: View {
    // 1. State management
    @StateObject private var recorder = AudioRecorder()
    @State private var transcription: String = "Press and hold to record..."
    @State private var isProcessing: Bool = false
    @State private var isModelLoading: Bool = true

    // ASR engine (protocol-based)
    @State private var currentEngine: (any ASREngineProtocol)?
    @AppStorage("selectedModel") private var selectedModelRaw = ASRModel.senseVoice.rawValue

    // DFX metrics
    @State private var latestMetrics: ASRSessionMetrics?
    @State private var showMetrics: Bool = false

    // Streaming
    @State private var isStreamingEnabled: Bool = false
    @State private var partialTranscription: String = ""
    @State private var isStreaming: Bool = false

    private var selectedModel: ASRModel {
        ASRModel(rawValue: selectedModelRaw) ?? .senseVoice
    }

    var body: some View {
        NavigationStack {
            VStack(spacing: 20) {
                // Model selector
                Picker("Model", selection: $selectedModelRaw) {
                    ForEach(ASRModel.allCases) { model in
                        Text(model.rawValue).tag(model.rawValue)
                    }
                }
                .pickerStyle(.segmented)
                .padding(.horizontal)

                // Streaming toggle
                Toggle("Streaming", isOn: $isStreamingEnabled)
                    .disabled(!selectedModel.supportsStreaming)
                    .padding(.horizontal)

                // Result display area
                if isModelLoading {
                    VStack {
                        ProgressView()
                            .scaleEffect(1.5)
                        Text("Loading \(selectedModel.rawValue) Model...")
                            .font(.caption)
                            .foregroundColor(.gray)
                            .padding(.top, 8)
                    }
                    .frame(maxHeight: .infinity)
                } else {
                    ScrollView {
                        VStack(alignment: .leading, spacing: 4) {
                            Text(transcription)
                                .font(.body)
                            if isStreaming && !partialTranscription.isEmpty {
                                Text(partialTranscription)
                                    .font(.body)
                                    .italic()
                                    .foregroundColor(.gray)
                            }
                        }
                        .padding()
                        .frame(maxWidth: .infinity, alignment: .leading)
                    }
                    .frame(maxHeight: .infinity)
                    .background(Color(UIColor.secondarySystemBackground))
                    .cornerRadius(12)
                    .padding(.horizontal)
                }

                // DFX overlay
                if showMetrics, let metrics = latestMetrics {
                    MetricsOverlayView(metrics: metrics)
                        .padding(.horizontal)
                }

                // Waveform view
                WaveformView(recorder: recorder)
                    .frame(height: 120)
                    .padding(.horizontal)

                Spacer()

                // Record button (press and hold)
                ZStack {
                    Circle()
                        .fill(recorder.isRecording ? Color.red : Color.blue)
                        .frame(width: 80, height: 80)
                        .scaleEffect(recorder.isRecording ? 1.2 : 1.0)
                        .opacity(recorder.isRecording ? 0.5 : 1.0)
                        .animation(.easeInOut(duration: 0.2), value: recorder.isRecording)

                    Image(systemName: recorder.isRecording ? "waveform" : "mic.fill")
                        .font(.system(size: 30))
                        .foregroundColor(.white)
                }
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
                .disabled(isProcessing || isModelLoading)
                .padding(.bottom, 30)

                if isProcessing {
                    ProgressView("Transcribing...")
                        .padding(.bottom, 10)
                }
            }
            .navigationTitle("\(selectedModel.rawValue) ASR")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button(action: { showMetrics.toggle() }) {
                        Image(systemName: showMetrics ? "chart.bar.fill" : "chart.bar")
                    }
                }
            }
        }
        .onAppear {
            recorder.requestPermission()
            initEngine(for: selectedModel)
        }
        .onChange(of: selectedModelRaw) { _ in
            let model = ASRModel(rawValue: selectedModelRaw) ?? .senseVoice
            // Reset streaming toggle if model doesn't support it
            if !model.supportsStreaming {
                isStreamingEnabled = false
            }
            initEngine(for: model)
        }
    }

    // MARK: - Logic

    func initEngine(for model: ASRModel) {
        isModelLoading = true
        currentEngine = nil
        transcription = "Press and hold to record..."
        latestMetrics = nil

        print("Initializing \(model.rawValue) Engine...")
        DispatchQueue.global(qos: .userInitiated).async {
            guard let resourcePath = Bundle.main.resourcePath else {
                DispatchQueue.main.async {
                    self.transcription = "Error: Could not find bundle resource path."
                    self.isModelLoading = false
                }
                return
            }

            let engine = ASREngineFactory.create(model: model)
            let success = engine.initialize(modelPath: resourcePath)

            DispatchQueue.main.async {
                if success {
                    self.currentEngine = engine
                    self.isModelLoading = false
                    print("\(model.rawValue) Engine Ready.")
                } else {
                    self.transcription = "Error: \(model.rawValue) model files not found or initialization failed."
                    self.isModelLoading = false
                }
            }
        }
    }

    func startRecording() {
        if isStreamingEnabled, let engine = currentEngine, engine.supportsStreaming {
            // Streaming mode
            isStreaming = true
            partialTranscription = ""
            engine.startStream()
            recorder.onChunkReady = { chunk in
                let data = chunk.withUnsafeBufferPointer { Data(buffer: $0) }
                if let partial = engine.feedChunk(data) {
                    DispatchQueue.main.async {
                        self.partialTranscription = partial
                    }
                }
            }
        }
        recorder.startRecording()
    }

    func stopRecordingAndTranscribe() {
        recorder.stopRecording()

        guard let engine = currentEngine else {
            transcription = "Error: Engine not initialized."
            return
        }

        if recorder.recordingData.count < 1600 {
            print("Audio too short.")
            if isStreaming {
                let _ = engine.endStream()
                isStreaming = false
                recorder.onChunkReady = nil
            }
            return
        }

        if isStreaming {
            // End streaming mode
            isProcessing = true
            DispatchQueue.global(qos: .userInitiated).async {
                let result = engine.endStream()
                DispatchQueue.main.async {
                    self.transcription = result.text.isEmpty ? "No text detected." : result.text
                    self.latestMetrics = result.metrics
                    self.isProcessing = false
                    self.isStreaming = false
                    self.partialTranscription = ""
                    self.recorder.onChunkReady = nil
                }
            }
        } else {
            // Batch mode
            isProcessing = true
            let rawData = recorder.recordingData

            DispatchQueue.global(qos: .userInitiated).async {
                let pcmData = rawData.withUnsafeBufferPointer { bufferPtr in
                    Data(buffer: bufferPtr)
                }

                let result = engine.transcribe(pcmData)

                DispatchQueue.main.async {
                    self.transcription = result.text.isEmpty ? "No text detected." : result.text
                    self.latestMetrics = result.metrics
                    self.isProcessing = false
                }
            }
        }
    }
}

struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}
