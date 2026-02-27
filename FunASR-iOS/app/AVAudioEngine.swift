//
//  AVAudioEngine.swift
//  FunASR-iOS
//
//  Created by Mingqian Yu on 1/14/26.
//

import Foundation
import AVFoundation

class AudioRecorder: NSObject, ObservableObject {
    @Published var isRecording = false
    @Published var recordingData: [Float] = [] // Stores the accumulated 16kHz PCM data

    // Streaming chunk callback: called when enough samples accumulate
    var onChunkReady: (([Float]) -> Void)?

    // Audio Engine Components
    private let audioEngine = AVAudioEngine()
    private var inputNode: AVAudioInputNode?
    private var converter: AVAudioConverter?

    // Whisper requires 16kHz
    private let targetSampleRate: Double = 16000.0

    // Streaming: accumulate samples and fire callback every streamChunkSize samples (300ms @ 16kHz)
    private let streamChunkSize: Int = 4800
    private var streamBuffer: [Float] = []
    
    func requestPermission() {
        if #available(iOS 17.0, *) {
            AVAudioApplication.requestRecordPermission { allowed in
                DispatchQueue.main.async {
                    if !allowed {
                        print("❌ Microphone permission denied.")
                    } else {
                        print("✅ Microphone permission granted.")
                    }
                }
            }
        } else {
            AVAudioSession.sharedInstance().requestRecordPermission { allowed in
                DispatchQueue.main.async {
                    if !allowed {
                        print("❌ Microphone permission denied.")
                    } else {
                        print("✅ Microphone permission granted.")
                    }
                }
            }
        }
    }
    
    /// Start Recording
    func startRecording() {
        // 1. Setup Audio Session (Play and Record mode)
        let session = AVAudioSession.sharedInstance()
        do {
            try session.setCategory(.playAndRecord, mode: .default, options: [.defaultToSpeaker, .allowBluetooth])
            try session.setActive(true)
        } catch {
            print("❌ Failed to setup audio session: \(error)")
            return
        }
        
        // 2. Clear previous data
        recordingData.removeAll()
        streamBuffer.removeAll()
        
        // 3. Setup Audio Engine
        inputNode = audioEngine.inputNode
        let inputFormat = inputNode!.inputFormat(forBus: 0)
        
        // Define the target format (16kHz, Mono, Float32)
        guard let outputFormat = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: targetSampleRate, channels: 1, interleaved: false) else {
            print("❌ Failed to create output format")
            return
        }
        
        // 4. Setup Converter (Input -> 16kHz)
        // Since the mic might be 48kHz, we need a converter.
        converter = AVAudioConverter(from: inputFormat, to: outputFormat)
        
        // 5. Install "Tap" (Intercept audio data from the mic)
        // bufferSize is just a hint, not a guarantee.
        inputNode!.installTap(onBus: 0, bufferSize: 1024, format: inputFormat) { [weak self] (buffer, time) in
            guard let self = self else { return }
            
            // Convert the captured buffer to 16kHz
            let inputCallback: AVAudioConverterInputBlock = { inNumPackets, outStatus in
                outStatus.pointee = .haveData
                return buffer
            }
            
            // Calculate how many frames we need for the output
            // Ratio = 16000 / 48000 = 1/3
            let ratio = self.targetSampleRate / inputFormat.sampleRate
            let capacity = UInt32(Double(buffer.frameLength) * ratio)
            
            guard let outputBuffer = AVAudioPCMBuffer(pcmFormat: outputFormat, frameCapacity: capacity) else { return }
            
            var error: NSError? = nil
            let status = self.converter?.convert(to: outputBuffer, error: &error, withInputFrom: inputCallback)
            
            if status == .error || error != nil {
                print("Error during conversion: \(error?.localizedDescription ?? "Unknown")")
                return
            }
            
            // 6. Extract float data and append to our array
            if let channelData = outputBuffer.floatChannelData {
                let channelPointer = channelData[0] // Mono, so we take channel 0
                let frameLength = Int(outputBuffer.frameLength)

                // Swift array from C pointer
                let samples = Array(UnsafeBufferPointer(start: channelPointer, count: frameLength))

                // Append to main array
                DispatchQueue.main.async {
                    self.recordingData.append(contentsOf: samples)
                }

                // Streaming chunk accumulation
                if self.onChunkReady != nil {
                    self.streamBuffer.append(contentsOf: samples)
                    while self.streamBuffer.count >= self.streamChunkSize {
                        let chunk = Array(self.streamBuffer.prefix(self.streamChunkSize))
                        self.streamBuffer.removeFirst(self.streamChunkSize)
                        let callback = self.onChunkReady
                        DispatchQueue.global(qos: .userInitiated).async {
                            callback?(chunk)
                        }
                    }
                }
            }
        }
        
        // 7. Start Engine
        do {
            try audioEngine.start()
            DispatchQueue.main.async { self.isRecording = true }
            print("🎤 Recording started...")
        } catch {
            print("❌ Could not start audio engine: \(error)")
        }
    }
    
    func stopRecording() {
        audioEngine.stop()
        inputNode?.removeTap(onBus: 0) // Remove the interceptor
        streamBuffer.removeAll()

        DispatchQueue.main.async {
            self.isRecording = false
        }
        print("Recording stopped. Captured \(recordingData.count) samples.")
    }
}
