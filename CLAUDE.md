# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

FunASR-iOS is an iOS speech recognition app with multiple ASR engines:
- **SenseVoice** (active) — non-autoregressive encoder-only model, CTC greedy decoding
- **Paraformer** (planned) — non-autoregressive, CIF predictor + parallel decoder, supports native streaming
- **Whisper** (legacy) — autoregressive encoder-decoder model

Inference runs via ONNX Runtime with CoreML execution provider (Apple Neural Engine).

## Build Commands

```bash
# Build for device
xcodebuild -project FunASR-iOS.xcodeproj -scheme FunASR-iOS -configuration Debug \
  -destination 'platform=iOS,name=<device_name>'

# Build for simulator
xcodebuild -project FunASR-iOS.xcodeproj -scheme FunASR-iOS -configuration Debug \
  -destination 'platform=iOS Simulator,name=iPhone 15'

# Run tests
xcodebuild test -project FunASR-iOS.xcodeproj -scheme FunASR-iOS \
  -destination 'platform=iOS Simulator,name=iPhone 15'

# Standalone C++ build (for testing inference outside Xcode)
cd FunASR-iOS/cpp_inference && cmake -B build && cmake --build build
```

Deployment targets: iOS 18.5 / macOS 15.4. C++ standard: GNU++20. Swift version: 5.0.

## Architecture

Three-layer architecture with strict separation:

```
Swift UI (SwiftUI)  →  Obj-C++ Bridge (.mm)  →  C++ Inference Engine
```

### Swift UI Layer (`FunASR-iOS/app/`)
- `ContentView.swift` — main recording/transcription UI
- `AVAudioEngine.swift` — `AudioRecorder` class, captures mic at 16kHz mono PCM
- `WaveformView.swift` — real-time waveform visualization

### Bridge Layer (`FunASR-iOS/bridge/`)
- `SenseVoiceContext.mm` — active bridge, converts Swift types ↔ C++ types
- `WhisperContext.mm` — legacy Whisper bridge
- Zero-copy data transfer: Swift `[Float]` → `NSData` → `const float*`

### C++ Inference Engine (`FunASR-iOS/cpp_inference/`)
- `sensevoice_engine.cpp` — single-pass inference, CTC decode, CMVN from model metadata
- `whisper_engine.cpp` — autoregressive encoder-decoder with beam search
- `feature_extractor.cpp` — Kaldi-compatible 80-dim log-mel filterbank (25ms frame, 10ms hop, 512-pt FFT). Uses Apple Accelerate (vDSP) for Hamming window and matrix ops
- `sensevoice_tokenizer.h` — SentencePiece tokenizer (vocab in `tokens.txt`)
- `whisper_tokenizer.cpp` — Whisper tokenizer (vocab in `vocab.json`, uses nlohmann/json)

### Models (`FunASR-iOS/cpp_inference/models/`)
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

---

## Roadmap: Paraformer 集成 + 流式 + DFX + 评测

### Phase 1 — 基础设施（ASR 协议抽象 + UI + DFX 框架）

#### 1.1 ASREngine 协议抽象

所有 ASR 模型遵守统一的 Swift protocol，通过工厂方法实例化：

```
ASREngineProtocol.swift   — 统一接口定义
ASREngineFactory.swift    — 根据 ASRModel enum 创建对应 engine
```

协议需包含：
- `initialize(modelPath:)` — 初始化模型
- `transcribe(_:)` — 批量推理（离线模式）
- `startStream()` / `feedChunk(_:)` / `endStream()` — 流式推理
- `supportsStreaming: Bool` — 能力声明

对应的 Obj-C++ Bridge 也需为每个模型提供独立的 Context 类：
- `SenseVoiceContext.mm` (已有)
- `ParaformerContext.mm` (新增)

#### 1.2 UI 改动

`ContentView.swift` 新增：
- **模型选择 Picker** — `ASRModel` enum（SenseVoice / Paraformer / 后续扩展），绑定 `@AppStorage`
- **流式输出 Toggle** — 模型不支持流式时自动灰化（`.disabled(!model.supportsStreaming)`）
- **实时转写文本区域** — 流式模式下通过回调持续更新 `@State var partialTranscription`

#### 1.3 DFX 采集框架

**采集指标：**

| 指标 | 定义 | 采集方法 |
|------|------|---------|
| 初始化时延 | `initialize()` 调用到 ready | `mach_absolute_time()` 前后打点 |
| 首字时延 | 音频结束 → 第一个字符输出 | 音频末帧时间戳 vs 第一个 token 时间戳 |
| 推理时延 | 单次 ONNX forward 耗时 | C++ 层 `std::chrono` 打点 |
| RTF（实时率） | inference_time / audio_duration | RTF < 1.0 = 实时，< 0.3 适合移动端 |
| 模型内存占用 | ONNX 加载后 RSS 增量 | `task_info(MACH_TASK_BASIC_INFO)` |
| 峰值内存 | 推理过程中最高 RSS | 推理线程中轮询采样 |
| 流式块延迟 | 每个 chunk 从输入到输出 | 每块独立打点（流式专属） |
| 热状态 | 设备降频风险 | `ProcessInfo.thermalState` 实时监听 |
| CPU 占用率 | 推理期间 CPU % | `task_threads` / Instruments |

**工具选型：**
- C++ 层：`mach_absolute_time()` 高精度计时 + `task_info()` 内存采集
- Swift 层：`os_signpost` 标记推理区间（配合 Instruments 可视化）
- 热状态：`ProcessInfo.thermalStateDidChangeNotification` 实时监听
- 功耗：iOS 无实时 mW API，用 RTF + 热状态 + CPU 占用间接评估；精确功耗需外接硬件电流计
- 数据结构：`ASRSessionMetrics` (Codable) 写入本地 JSON，测试后导出

### Phase 2 — SenseVoice 伪流式（蹦字）

SenseVoice 是非自回归模型，无法原生流式。采用 **VAD 触发 + 定时兜底** 组合方案：

- 检测到停顿 > 300ms → 立即触发一次完整推理（自然断句点蹦字）
- 持续说话超过 3s 无停顿 → 强制触发一次推理
- 显示区分"已确认段"和"推理中段"

**VAD 选型：** 优先使用能量阈值简易 VAD（`RMS < threshold` 持续 200ms = 静音），后续可升级为 Silero VAD (ONNX)。

**改动文件：**
- `AVAudioEngine.swift` — 增加分块触发逻辑（每 N 帧检测 VAD → 回调 bridge 层）
- `SenseVoiceContext.mm` — 新增 `feedChunk:callback:` 方法（内部仍是累积后完整推理）

### Phase 3 — Paraformer 接入

#### Paraformer Pipeline（与 SenseVoice 对比）

```
SenseVoice:  PCM → Fbank → LFR(7,6) → CMVN → Encoder → CTC argmax → detokenize
Paraformer:  PCM → Fbank → CMVN → Encoder(分块+cache) → CIF Predictor → Parallel Decoder → detokenize
```

关键差异：
- **无 LFR**：Paraformer 直接用 80-dim fbank，`feature_extractor.cpp` 可复用
- **CIF 解码**：Predictor 预测 token 数量，再触发并行 decoder（比 CTC argmax 复杂得多）
- **流式 Cache**：Paraformer-streaming 的 Encoder 每次 forward 需传入/更新 cache 张量
- **CMVN**：外部文件 `am.mvn`（SenseVoice 嵌入 ONNX metadata）
- **Tokenizer**：同为 SentencePiece，但 vocab 不同（Paraformer ~8k 中文字符 vs SenseVoice 25055）

**新增文件：**
- `paraformer_engine.h / .cpp` — C++ 推理引擎（~600-800 行，含 cache 管理 + CIF 解码）
- `ParaformerContext.h / .mm` — Obj-C++ Bridge（流式接口）

**前置依赖：** 需先从 FunASR 导出 Paraformer-streaming ONNX 模型，模型的 input/output tensor 定义决定 C++ engine 实现细节。

### Phase 4 — 评测

**评测工具：** 独立 Python 脚本（不在 iOS 上跑），用 `jiwer` 计算 CER/WER。

**重点测评集：** MagicData-RAMC（180h 中文电话对话，含底噪/口音/自然停顿）。

**评测维度：**
- CER（字错误率）— 中文核心指标
- RTF — 推理速度
- 底噪鲁棒性 — 添加 0dB/5dB/10dB/20dB SNR 底噪
- SER（句子错误率）— 通话场景整句准确率

**已知基准（来自公开论文，需核实）：**

| 模型 | MagicData-RAMC CER | 备注 |
|------|-------------------|------|
| Paraformer-zh (large, offline) | ~9–11% | FunASR benchmark |
| Paraformer-zh-streaming | ~12–15% | 流式，延迟与精度 tradeoff |
| SenseVoice-Small | ~13–16% | 多任务模型 |
| SenseVoice-Large | ~10–13% | 较 Small 有提升 |

参考论文：SenseVoice (arXiv:2407.04051), Paraformer (arXiv:2206.08317)
