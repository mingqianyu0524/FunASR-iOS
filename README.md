# FunASR-iOS

[![CI](https://github.com/mingqianyu0524/FunASR-iOS/actions/workflows/ci.yml/badge.svg)](https://github.com/mingqianyu0524/FunASR-iOS/actions/workflows/ci.yml)

iOS speech recognition app powered by [FunASR](https://github.com/modelscope/FunASR) models running on-device via ONNX Runtime with CoreML (Apple Neural Engine) acceleration.

## Models

| Model | Status | Description |
|-------|--------|-------------|
| **SenseVoice** | ✅ Active | Non-autoregressive encoder-only, CTC decoding, multilingual |
| **Paraformer** | ✅ Active | Non-autoregressive CIF + parallel decoder, native streaming |
| Whisper | Legacy | Autoregressive encoder-decoder |

All models run as INT8-quantized ONNX graphs. Model files are distributed via [GitHub Release `models-v1.0`](https://github.com/mingqianyu0524/FunASR-iOS/releases/tag/models-v1.0).

## Demo

https://github.com/mingqianyu0524/FunASR-iOS/releases/download/demo-v1.0/sensevoice_demo.MOV

## Accuracy Benchmarks

Evaluated with `scripts/eval_python.py` using FunASR Python + ONNX Runtime on CPU (2026-03-05).

| Model | Dataset | Utterances | CER | SER |
|-------|---------|------------|-----|-----|
| SenseVoice INT8 | AISHELL-1 test | 7,176 | **13.97%** | 99.87% |
| SenseVoice INT8 | RAMC sample | 500 | **14.81%** | 57.2% |
| Paraformer INT8 | AISHELL-1 test | 7,176 | **2.28%** | 22.4% |
| Paraformer INT8 | RAMC sample | 25 | **7.83%** | 48.0% |

- **AISHELL-1**: read speech, studio conditions (OpenSLR #33)
- **RAMC**: conversational speech, varied acoustic conditions (MagicData-RAMC, OpenSLR #123)

CI runs `testBatchEval()` + `testParaformerBatchEval()` on 25+25 fixture utterances and reports delta vs these baselines in the [GitHub Actions step summary](https://github.com/mingqianyu0524/FunASR-iOS/actions). Full baseline JSONs in [`eval_results/`](eval_results/).

## Architecture

```
Swift UI (SwiftUI)  ──▶  Obj-C++ Bridge (.mm)  ──▶  C++ Inference Engine
```

| Layer | Path | Responsibility |
|-------|------|----------------|
| Swift UI | `FunASR-iOS/app/` | Recording, waveform, transcription display |
| Bridge | `FunASR-iOS/bridge/SenseVoiceContext.mm` | Swift ↔ C++ type conversion, zero-copy |
| C++ Engine | `FunASR-iOS/cpp_inference/` | Feature extraction, ONNX inference, tokenization |

**SenseVoice pipeline:** 16kHz PCM → 80-dim log-mel filterbank (25ms/10ms) → LFR (7-frame window, 6-frame shift) → CMVN → ONNX `[1, T, 560]` → CTC logits → greedy decode → SentencePiece detokenize

Feature extraction uses Apple Accelerate (vDSP) for Hamming window and matrix operations.

## Requirements

- Xcode 16.4+
- iOS 18.5+ deployment target
- ONNX Runtime 1.21.0 xcframework (download separately, see Setup)
- Model files from `models-v1.0` release (download separately, see Setup)

## Setup

**1. Clone and open in Xcode**

```bash
git clone https://github.com/mingqianyu0524/FunASR-iOS.git
open FunASR-iOS.xcodeproj
```

**2. Download ONNX Runtime xcframework**

```bash
ORT_VERSION=1.21.0
curl -L "https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/onnxruntime-ios-xcframework-${ORT_VERSION}.zip" \
  -o ort.zip
unzip ort.zip -d FunASR-iOS/cpp_inference/lib/
```

**3. Download model files**

```bash
mkdir -p FunASR-iOS/cpp_inference/models
gh release download models-v1.0 \
  --repo mingqianyu0524/FunASR-iOS \
  --dir FunASR-iOS/cpp_inference/models/
```

**4. Build and run**

Select an iOS Simulator or device target in Xcode and press ▶.

## CI

The GitHub Actions workflow (`.github/workflows/ci.yml`) runs on every push to `main`:

1. Downloads ORT xcframework (cached by version)
2. Stages model files from `models-v1.0` release
3. Downloads CI accuracy fixtures from `test-fixtures-v1.0` release (25 RAMC + 25 AISHELL-1 utterances)
4. Builds and runs `xcodebuild test` on iPhone 16 iOS 18.5 Simulator
5. Runs `testSenseVoiceSample` (single RAMC utterance, CER < 20%) and `testBatchEval` (25+25 fixtures, CER < 30%)
6. Generates accuracy report with delta vs baselines in the step summary

## Evaluation

To reproduce the Python baseline CER numbers:

```bash
pip install funasr onnxruntime jiwer tqdm

# AISHELL-1 (requires ~15GB dataset from openslr.org/33)
python3 scripts/eval_python.py \
  --model sensevoice \
  --model-dir /path/to/models \
  --audio-dir /path/to/data_aishell/wav/test \
  --transcript /path/to/data_aishell/transcript/aishell_transcript_v0.8.txt \
  --output eval_results/sensevoice_aishell1.json

# MagicData-RAMC (requires ~15GB dataset from openslr.org/123)
# Uses two-pass streaming — see scripts/ for details
```

Baseline results are stored in `eval_results/`. CI compares against these to detect regressions.

## License

GPL-3.0. See [LICENSE](LICENSE).
