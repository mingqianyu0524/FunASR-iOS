# FunASR-iOS

An iOS speech recognition application featuring multiple ASR (Automatic Speech Recognition) engines running on-device with Apple Neural Engine acceleration.

## Features

- **Multiple ASR Engines**
  - **SenseVoice** — Non-autoregressive encoder-only model with CTC decoding
  - **Paraformer** — Non-autoregressive model with CIF predictor and parallel decoder
- **On-Device Inference** — ONNX Runtime with CoreML execution provider (ANE acceleration)
- **Real-time Audio Processing** — 16kHz mono PCM capture with live waveform visualization
- **DFX Metrics** — Real-time inference latency, RTF, memory usage overlay
- **Optimized Performance** — INT8 quantization, zero-copy bridging, memory-efficient inference

## Architecture

FunASR-iOS uses a three-layer architecture with strict separation of concerns:

```
┌─────────────────────────────────────────┐
│   Swift UI Layer (SwiftUI)             │
│   - Recording interface                │
│   - Model selection (SenseVoice/Para.) │
│   - Waveform visualization             │
│   - Transcription display              │
│   - DFX metrics overlay               │
└──────────────┬──────────────────────────┘
               │ NSData (zero-copy)
┌──────────────▼──────────────────────────┐
│   Bridge Layer (Objective-C++)         │
│   - SenseVoiceContext / ParaformerCtx  │
│   - Type conversion Swift ↔ C++        │
│   - Exception handling                 │
└──────────────┬──────────────────────────┘
               │ std::vector<float>
┌──────────────▼──────────────────────────┐
│   C++ Inference Engine                 │
│   - Kaldi-compatible feature extraction│
│   - ONNX Runtime inference             │
│   - CTC / CIF+parallel decoding       │
└─────────────────────────────────────────┘
```

### SenseVoice Pipeline

```
Audio (16kHz PCM)
  → 80-dim log-mel Fbank (25ms window, 10ms hop, 512-pt FFT)
  → LFR (7-frame window, 6-frame shift, ~85% length reduction)
  → CMVN normalization (from model metadata)
  → ONNX Runtime inference [1, T_lfr, 560]
  → CTC greedy decode → SentencePiece detokenize
```

### Paraformer Pipeline

```
Audio (16kHz PCM)
  → 80-dim log-mel Fbank (25ms window, 10ms hop, 512-pt FFT)
  → CMVN normalization (from am.mvn)
  → Encoder ONNX → CIF Predictor → Parallel Decoder ONNX
  → Token decode → detokenize
```

## Requirements

- **Platform:** iOS 18.5+ / macOS 15.4+
- **Devices:** iPhone, iPad, Apple Vision Pro
- **Development:** Xcode 16+
- **Build Tools:** Swift 5.0, C++20

## Setup

### 1. Clone and checkout

```bash
git clone https://github.com/mingqianyu0524/FunASR-iOS.git
cd FunASR-iOS
```

### 2. Download ONNX Runtime

```bash
curl -L -o ort.zip \
  https://download.onnxruntime.ai/pod-archive-onnxruntime-c-1.21.0.zip
unzip -o ort.zip
mkdir -p FunASR-iOS/cpp_inference/lib/
mv onnxruntime.xcframework FunASR-iOS/cpp_inference/lib/onnxruntime.xcframework
rm ort.zip
```

### 3. Download model files

```bash
# Download from GitHub Releases
gh release download models-v1.0 \
  --repo mingqianyu0524/FunASR-iOS \
  --dir ci_models/

# Stage with names the Xcode project expects
mkdir -p FunASR-iOS/cpp_inference/models/

# SenseVoice (names match directly)
cp ci_models/sensevoice.int8.onnx     FunASR-iOS/cpp_inference/models/
cp ci_models/sensevoice_tokens.txt     FunASR-iOS/cpp_inference/models/

# Paraformer (rename enc/dec → encoder/decoder)
cp ci_models/paraformer_enc.int8.onnx  FunASR-iOS/cpp_inference/models/paraformer_encoder.int8.onnx
cp ci_models/paraformer_dec.int8.onnx  FunASR-iOS/cpp_inference/models/paraformer_decoder.int8.onnx
cp ci_models/paraformer_am.mvn         FunASR-iOS/cpp_inference/models/am.mvn

# Convert paraformer tokens txt → JSON array
python3 -c "
import json
tokens = [l.split()[0] for l in open('ci_models/paraformer_tokens.txt') if l.strip()]
open('FunASR-iOS/cpp_inference/models/paraformer_tokens.json','w').write(json.dumps(tokens, ensure_ascii=False))
print(f'Converted {len(tokens)} tokens to JSON')
"

rm -rf ci_models/
```

### 4. Build

```bash
# Build for simulator
xcodebuild -project FunASR-iOS.xcodeproj \
  -scheme FunASR-iOS \
  -destination 'platform=iOS Simulator,name=iPhone 16' \
  CODE_SIGN_IDENTITY="" CODE_SIGNING_REQUIRED=NO

# Run tests
xcodebuild test -project FunASR-iOS.xcodeproj \
  -scheme FunASR-iOS \
  -destination 'platform=iOS Simulator,name=iPhone 16' \
  CODE_SIGN_IDENTITY="" CODE_SIGNING_REQUIRED=NO
```

Or open `FunASR-iOS.xcodeproj` in Xcode and press ⌘R.

## Project Structure

```
FunASR-iOS/
├── app/                              # Swift UI layer
│   ├── FunASR_iOSApp.swift          # App entry point
│   ├── ContentView.swift            # Main UI (recording/transcription/model picker)
│   ├── AVAudioEngine.swift          # Audio recording (16kHz PCM)
│   ├── WaveformView.swift           # Real-time waveform visualization
│   ├── MetricsOverlayView.swift     # DFX metrics overlay
│   ├── ASREngineProtocol.swift      # Unified ASR engine protocol
│   ├── ASREngineFactory.swift       # Engine factory (SenseVoice/Paraformer)
│   ├── SenseVoiceASREngine.swift    # SenseVoice engine wrapper
│   ├── ParaformerASREngine.swift    # Paraformer engine wrapper
│   └── ASRSessionMetrics.swift      # Inference metrics collection
│
├── bridge/                           # Objective-C++ bridge layer
│   ├── SenseVoiceContext.mm         # SenseVoice bridge
│   ├── ParaformerContext.mm         # Paraformer bridge
│   └── ASRBridgeResult.mm          # Shared result type
│
├── cpp_inference/                    # C++ inference engine
│   ├── include/
│   │   ├── sensevoice_engine.h
│   │   ├── paraformer_engine.h
│   │   ├── feature_extractor.h      # 80-dim log-Fbank (Kaldi-compatible)
│   │   └── sensevoice_tokenizer.h   # SentencePiece tokenizer
│   │
│   ├── src/
│   │   ├── sensevoice_engine.cpp
│   │   ├── paraformer_engine.cpp
│   │   └── feature_extractor.cpp
│   │
│   ├── lib/
│   │   └── onnxruntime.xcframework  # ONNX Runtime v1.21.0 (gitignored)
│   │
│   └── models/                       # ONNX models (gitignored)
│       ├── sensevoice.int8.onnx          # SenseVoice INT8 (~228 MB)
│       ├── sensevoice_tokens.txt         # SenseVoice vocabulary (25,055 tokens)
│       ├── paraformer_encoder.int8.onnx  # Paraformer encoder INT8 (~158 MB)
│       ├── paraformer_decoder.int8.onnx  # Paraformer decoder INT8 (~68 MB)
│       ├── am.mvn                        # Paraformer CMVN stats
│       └── paraformer_tokens.json        # Paraformer vocabulary (8,404 tokens)
│
└── Assets.xcassets/                  # App icons and images
```

## Models

Model files are not included in the repository due to size constraints (gitignored). See [Setup](#setup) for download instructions.

### SenseVoice
- **Model:** `sensevoice.int8.onnx` (~228 MB, INT8 quantized)
- **Vocabulary:** `sensevoice_tokens.txt` (~316 KB, 25,055 tokens)
- **Type:** Non-autoregressive encoder-only
- **Decoding:** CTC greedy decode

### Paraformer
- **Encoder:** `paraformer_encoder.int8.onnx` (~158 MB, INT8 quantized)
- **Decoder:** `paraformer_decoder.int8.onnx` (~68 MB, INT8 quantized)
- **CMVN:** `am.mvn` (~11 KB)
- **Vocabulary:** `paraformer_tokens.json` (~94 KB, 8,404 tokens)
- **Type:** Non-autoregressive with CIF predictor + parallel decoder

## Dependencies

### Native iOS Frameworks
- `SwiftUI` — Declarative UI framework
- `AVFoundation` — Audio recording and processing
- `Accelerate.framework` — SIMD/vDSP optimizations (FFT, matrix ops, windowing)
- `CoreML.framework` — Neural Engine acceleration

### Third-Party Libraries
- **ONNX Runtime v1.21.0** (XCFramework) — ML inference with CoreML execution provider
- **dr_wav.h** — Header-only WAV file loader
- **nlohmann/json.hpp** — Header-only JSON parser

## License

[Add your license information here]

## Acknowledgments

- [FunASR](https://github.com/modelscope/FunASR) — SenseVoice and Paraformer model architectures
- [ONNX Runtime](https://onnxruntime.ai/) — Cross-platform ML inference
