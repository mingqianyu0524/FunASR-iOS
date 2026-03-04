# FunASR-iOS

On-device Chinese ASR for iOS using SenseVoice and Paraformer INT8 ONNX models, accelerated via ONNX Runtime + CoreML (Apple Neural Engine).

## Accuracy Benchmarks

Baselines measured with FunASR Python + ONNX INT8 models (2026-03-04).

| Model | Dataset | Utterances | CER | SER |
|-------|---------|------------|-----|-----|
| SenseVoice INT8 | AISHELL-1 test | 7,176 | **13.97%** | 99.87% |
| SenseVoice INT8 | RAMC sample | 500 | **14.81%** | 57.2% |
| Paraformer INT8 | AISHELL-1 test | — | pending | — |
| Paraformer INT8 | RAMC test | — | pending | — |

- **AISHELL-1**: read speech, studio conditions (OpenSLR #33)
- **RAMC**: conversational speech, varied acoustic conditions (MagicData-RAMC, OpenSLR #123)
- CER = Character Error Rate, SER = Sentence Error Rate

CI accuracy regression tests run against 25+25 fixture samples from these datasets. Full baseline JSONs in [`eval_results/`](eval_results/).

## Architecture

Three-layer design: SwiftUI → Objective-C++ bridge → C++ ONNX inference engine.

```
SwiftUI (recording UI, model picker, DFX metrics)
    ↓ NSData (zero-copy)
Objective-C++ bridge (SenseVoiceContext / ParaformerContext)
    ↓ std::vector<float>
C++ engine (Fbank → LFR/CMVN → ONNX Runtime → CTC/CIF decode)
```

See [`FunASR-iOS/README.md`](FunASR-iOS/README.md) for full architecture diagrams, pipeline details, and project structure.

## Quick Start

```bash
git clone https://github.com/mingqianyu0524/FunASR-iOS.git
cd FunASR-iOS

# 1. Download ONNX Runtime
curl -L -o ort.zip https://download.onnxruntime.ai/pod-archive-onnxruntime-c-1.21.0.zip
unzip -o ort.zip && mkdir -p FunASR-iOS/cpp_inference/lib/
mv onnxruntime.xcframework FunASR-iOS/cpp_inference/lib/ && rm ort.zip

# 2. Download models from GitHub Releases
gh release download models-v1.0 --repo mingqianyu0524/FunASR-iOS --dir ci_models/
# (stage models per FunASR-iOS/README.md setup instructions)

# 3. Build & test
xcodebuild test -project FunASR-iOS.xcodeproj -scheme FunASR-iOS \
  -destination 'platform=iOS Simulator,name=iPhone 16' \
  CODE_SIGN_IDENTITY="" CODE_SIGNING_REQUIRED=NO
```

Full setup instructions: [`FunASR-iOS/README.md`](FunASR-iOS/README.md#setup)

## CI

GitHub Actions runs on every push to `main`. Each run:
1. Downloads ONNX Runtime + model files (cached)
2. Downloads CI fixture audio samples from `test-fixtures-v1.0` release (25 RAMC + 25 AISHELL-1)
3. Builds and runs `testBatchEval()` on iOS Simulator
4. Generates accuracy report vs baselines in [`eval_results/`](eval_results/) and posts to Actions Step Summary

## Models

| Model | File | Size | Notes |
|-------|------|------|-------|
| SenseVoice | `sensevoice.int8.onnx` | ~228 MB | Encoder-only, CTC decode |
| Paraformer encoder | `paraformer_encoder.int8.onnx` | ~158 MB | CIF predictor |
| Paraformer decoder | `paraformer_decoder.int8.onnx` | ~68 MB | Parallel decode |

Models are gitignored. Download via `gh release download models-v1.0`.

## License

GPLv3 — see [LICENSE](LICENSE).

## Acknowledgments

- [FunASR](https://github.com/modelscope/FunASR) — SenseVoice and Paraformer architectures
- [ONNX Runtime](https://onnxruntime.ai/) — ML inference runtime
