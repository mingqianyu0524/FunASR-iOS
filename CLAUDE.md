# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

WhisperIOS is an iOS speech recognition app with two ASR engines:
- **SenseVoice** (active) — non-autoregressive encoder-only model, CTC greedy decoding
- **Whisper** (legacy) — autoregressive encoder-decoder model

Inference runs via ONNX Runtime with CoreML execution provider (Apple Neural Engine).

## Build Commands

```bash
# Build for device
xcodebuild -project WhisperIOS.xcodeproj -scheme WhisperIOS -configuration Debug \
  -destination 'platform=iOS,name=<device_name>'

# Build for simulator
xcodebuild -project WhisperIOS.xcodeproj -scheme WhisperIOS -configuration Debug \
  -destination 'platform=iOS Simulator,name=iPhone 15'

# Run tests
xcodebuild test -project WhisperIOS.xcodeproj -scheme WhisperIOS \
  -destination 'platform=iOS Simulator,name=iPhone 15'

# Standalone C++ build (for testing inference outside Xcode)
cd WhisperIOS/cpp_inference && cmake -B build && cmake --build build
```

Deployment targets: iOS 18.5 / macOS 15.4. C++ standard: GNU++20. Swift version: 5.0.

## Architecture

Three-layer architecture with strict separation:

```
Swift UI (SwiftUI)  →  Obj-C++ Bridge (.mm)  →  C++ Inference Engine
```

### Swift UI Layer (`WhisperIOS/app/`)
- `ContentView.swift` — main recording/transcription UI
- `AVAudioEngine.swift` — `AudioRecorder` class, captures mic at 16kHz mono PCM
- `WaveformView.swift` — real-time waveform visualization

### Bridge Layer (`WhisperIOS/bridge/`)
- `SenseVoiceContext.mm` — active bridge, converts Swift types ↔ C++ types
- `WhisperContext.mm` — legacy Whisper bridge
- Zero-copy data transfer: Swift `[Float]` → `NSData` → `const float*`

### C++ Inference Engine (`WhisperIOS/cpp_inference/`)
- `sensevoice_engine.cpp` — single-pass inference, CTC decode, CMVN from model metadata
- `whisper_engine.cpp` — autoregressive encoder-decoder with beam search
- `feature_extractor.cpp` — Kaldi-compatible 80-dim log-mel filterbank (25ms frame, 10ms hop, 512-pt FFT). Uses Apple Accelerate (vDSP) for Hamming window and matrix ops
- `sensevoice_tokenizer.h` — SentencePiece tokenizer (vocab in `tokens.txt`)
- `whisper_tokenizer.cpp` — Whisper tokenizer (vocab in `vocab.json`, uses nlohmann/json)

### Models (`WhisperIOS/cpp_inference/models/`)
All `.onnx` and `.bin` files are gitignored. Key models:
- `model.int8.onnx` (228 MB) — SenseVoice INT8 quantized
- `encoder.onnx` / `decoder.onnx` — Whisper FP32
- `encoder_int8.onnx` / `decoder_int8.onnx` — Whisper INT8

### SenseVoice Pipeline Details
Audio → 80-mel Fbank → LFR (7-frame window, 6-frame shift, reduces length ~85%) → CMVN → ONNX model `[1, T_lfr, 560]` → CTC logits `[1, T, 25055]` → greedy decode → SentencePiece detokenize

## Key Dependencies

- **ONNX Runtime** — xcframework at `cpp_inference/lib/onnxruntime.xcframework` (gitignored)
- **Apple Accelerate** — SIMD/vDSP for feature extraction
- **CoreML.framework** — Neural Engine acceleration
- **dr_wav.h** — header-only WAV loader
- **nlohmann/json.hpp** — header-only JSON parser (Whisper tokenizer)

## Model Export

```bash
python3 export_fp16_without_emb.py  # Exports Whisper to ONNX FP16
# Requires: torch, whisper, onnx, onnxconverter-common
```

## Notes

- Comments in C++ files are mixed Chinese and English
- No linting or formatting tools are configured
- Tests are minimal stubs (Swift Testing + XCTest)
- The ONNX Runtime xcframework and all model files are gitignored — they must be obtained separately
