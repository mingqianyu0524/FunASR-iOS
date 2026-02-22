# SenseVoiceIOS

An iOS speech recognition application featuring two state-of-the-art ASR (Automatic Speech Recognition) engines running on-device with Apple Neural Engine acceleration.

## Features

- **Dual ASR Engines**
  - **SenseVoice** (active) — Non-autoregressive encoder-only model with CTC decoding
  - **Whisper** (legacy) — OpenAI's autoregressive encoder-decoder model
- **On-Device Inference** — ONNX Runtime with CoreML execution provider (ANE acceleration)
- **Real-time Audio Processing** — 16kHz mono PCM capture with live waveform visualization
- **Optimized Performance** — INT8 quantization, zero-copy bridging, memory-efficient inference

## Architecture

WhisperIOS uses a three-layer architecture with strict separation of concerns:

```
┌─────────────────────────────────────────┐
│   Swift UI Layer (SwiftUI)             │
│   - Recording interface                │
│   - Waveform visualization             │
│   - Transcription display               │
└──────────────┬──────────────────────────┘
               │ NSData (zero-copy)
┌──────────────▼──────────────────────────┐
│   Bridge Layer (Objective-C++)         │
│   - Type conversion Swift ↔ C++        │
│   - Exception handling                 │
└──────────────┬──────────────────────────┘
               │ std::vector<float>
┌──────────────▼──────────────────────────┐
│   C++ Inference Engine                 │
│   - Kaldi-compatible feature extraction│
│   - ONNX Runtime inference             │
│   - CTC/beam search decoding           │
└─────────────────────────────────────────┘
```

### SenseVoice Pipeline

```
Microphone (48kHz)
  → AVAudioEngine resample (16kHz mono)
  → Feature Extraction:
      • 25ms Hamming window, 10ms hop
      • 512-point FFT
      • 80-dimensional mel filterbank
      • Natural log transform
      • LFR (7-frame window, 6-frame shift)
      • CMVN normalization
  → ONNX Runtime (INT8 model, 228 MB)
  → CTC Greedy Decode
  → SentencePiece Detokenization
  → Transcription Result
```

## Requirements

- **Platform:** iOS 18.5+ / macOS 15.4+
- **Devices:** iPhone, iPad, Apple Vision Pro
- **Development:** Xcode 18.5+
- **Build Tools:** Swift 5.0, C++20

## Build Instructions

### Using Xcode (Recommended)

1. Open `WhisperIOS.xcodeproj` in Xcode
2. Select target device or simulator
3. Build and run (⌘R)

### Command Line

```bash
# Build for device
xcodebuild -project WhisperIOS.xcodeproj \
  -scheme WhisperIOS \
  -configuration Debug \
  -destination 'platform=iOS,name=<your-device-name>'

# Build for simulator
xcodebuild -project WhisperIOS.xcodeproj \
  -scheme WhisperIOS \
  -configuration Debug \
  -destination 'platform=iOS Simulator,name=iPhone 15'

# Run tests
xcodebuild test -project WhisperIOS.xcodeproj \
  -scheme WhisperIOS \
  -destination 'platform=iOS Simulator,name=iPhone 15'
```

### Standalone C++ Build (Testing Only)

```bash
cd WhisperIOS/cpp_inference
cmake -B build
cmake --build build
```

## Project Structure

```
WhisperIOS/
├── app/                          # Swift UI layer
│   ├── WhisperIOSApp.swift      # App entry point
│   ├── ContentView.swift        # Main UI (recording/transcription)
│   ├── AVAudioEngine.swift      # Audio recording (16kHz PCM)
│   └── WaveformView.swift       # Real-time waveform visualization
│
├── bridge/                       # Objective-C++ bridge layer
│   ├── SenseVoiceContext.mm     # Active: SenseVoice bridge
│   └── WhisperContext.mm        # Legacy: Whisper bridge
│
├── cpp_inference/               # C++ inference engine
│   ├── include/
│   │   ├── sensevoice_engine.h      # SenseVoice inference
│   │   ├── whisper_engine.h         # Whisper inference
│   │   ├── feature_extractor.h      # 80-dim log-Fbank extraction
│   │   ├── sensevoice_tokenizer.h   # SentencePiece tokenizer
│   │   └── whisper_tokenizer.h      # Whisper tokenizer
│   │
│   ├── src/                     # Implementation files
│   │   ├── sensevoice_engine.cpp
│   │   ├── whisper_engine.cpp
│   │   ├── feature_extractor.cpp
│   │   └── whisper_tokenizer.cpp
│   │
│   ├── lib/
│   │   └── onnxruntime.xcframework  # ONNX Runtime (gitignored)
│   │
│   └── models/                  # ONNX models (gitignored)
│       ├── model.int8.onnx          # SenseVoice INT8 (228 MB)
│       ├── tokens.txt               # SenseVoice vocabulary
│       ├── encoder.onnx             # Whisper encoder FP32
│       ├── decoder.onnx             # Whisper decoder FP32
│       ├── encoder_int8.onnx        # Whisper encoder INT8
│       ├── decoder_int8.onnx        # Whisper decoder INT8
│       └── vocab.json               # Whisper vocabulary
│
└── Assets.xcassets/             # App icons and images
```

## Models

**Note:** Model files are not included in the repository due to size constraints (gitignored).

### SenseVoice (Active)
- **Model:** `model.int8.onnx` (228 MB, INT8 quantized)
- **Vocabulary:** `tokens.txt` (~316 KB, 25,055 tokens)
- **Type:** Non-autoregressive encoder-only
- **Decoding:** CTC greedy decode
- **Languages:** Primarily Chinese (language ID: 3)

### Whisper (Legacy)
- **Encoder:** `encoder.onnx` (31 MB, FP32) or `encoder_int8.onnx` (9.6 MB)
- **Decoder:** `decoder.onnx` (190 MB, FP32) or `decoder_int8.onnx` (49 MB)
- **Vocabulary:** `vocab.json` (1.1 MB) + `vocab_embedding_fp32.bin` (152 MB)
- **Type:** Autoregressive encoder-decoder
- **Decoding:** Beam search

### Model Export

To export Whisper models to ONNX format:

```bash
python3 export_fp16_without_emb.py
```

**Requirements:** `torch`, `whisper`, `onnx`, `onnxconverter-common`

## Dependencies

### Native iOS Frameworks
- `SwiftUI` — Declarative UI framework
- `AVFoundation` — Audio recording and processing
- `Accelerate.framework` — SIMD/vDSP optimizations
- `CoreML.framework` — Neural Engine acceleration

### Third-Party Libraries
- **ONNX Runtime** (XCFramework) — Cross-platform ML inference with CoreML provider
- **dr_wav.h** — Header-only WAV file loader
- **nlohmann/json.hpp** — Header-only JSON parser

## Performance Optimizations

1. **INT8 Quantization** — 4× model size reduction with minimal accuracy loss
2. **CoreML Execution Provider** — Hardware acceleration via Apple Neural Engine
3. **Zero-Copy Bridging** — Direct pointer access between Swift and C++
4. **LFR (Low Frame Rate)** — ~85% sequence length reduction (7-frame window, 6-frame shift)
5. **Memory Management** — Device allocator prevents ONNX Runtime memory accumulation
6. **Accelerate Framework** — Apple vDSP for optimized FFT, matrix operations, windowing

## Known Issues

- Tests are minimal (placeholder stubs only)
- No linting or code formatting configuration
- ONNX Runtime xcframework and model files must be obtained separately
- Comments in C++ files are mixed Chinese/English

## License

[Add your license information here]

## Acknowledgments

- OpenAI Whisper for the original Whisper model architecture
- SenseVoice team for the non-autoregressive ASR model
- ONNX Runtime for cross-platform inference support
