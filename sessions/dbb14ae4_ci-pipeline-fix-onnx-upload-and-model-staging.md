# Session dbb14ae4 — CI pipeline fix: ORT URL / model staging / Xcode version

**日期：** 2026-03-01
**分支：** claude/pipeline-setup-isL66
**结果：** CI 运行到 "Build and run pipeline tests" 步骤，exit code 65（测试跑起来了但断言失败），留待下一 session 修复

---

## 背景

上一 session 把 Paraformer ONNX 模型导出并上传到了 GitHub Release `models-v1.0`。本 session 任务是修复 `claude/pipeline-setup-isL66` 分支上的 CI 问题。

---

## 问题 1：ONNX Runtime 下载 URL 失效

**现象：** `unzip` exit code 9，文件只有 9 字节。

**原因：** URL 写的是 `onnxruntime-objc-1.20.1.zip`，该文件在 GitHub Releases 上不存在（ORT 没有独立 iOS xcframework 的 GitHub Release 包）。

**修复：**
- 改用 CocoaPods pod archive：`https://download.onnxruntime.ai/pod-archive-onnxruntime-c-1.21.0.zip`（200 OK，45.9 MB）
- zip 解压后 `onnxruntime.xcframework` 在顶层，含 ios-arm64 / ios-arm64_x86_64-simulator / macos-arm64_x86_64 三个 slice
- 加 `mkdir -p FunASR-iOS/cpp_inference/lib/`
- cache key 改为 `onnxruntime-1.21.0`

---

## 问题 2：模型文件名不匹配

**现象：** `paraformer_encoder.int8.onnx: No such file or directory`（Xcode Bundle Resources 阶段）

**原因链：**
- GitHub Release 资产名：`sensevoice.int8.onnx`, `sensevoice_tokens.txt`, `paraformer_enc.int8.onnx`, …
- Xcode 项目 Bundle Resources 期待内部名：`model.int8.onnx`, `tokens.txt`, `paraformer_encoder.int8.onnx`, `paraformer_decoder.int8.onnx`, `am.mvn`, `paraformer_tokens.json`
- C++ 引擎路径拼接也使用内部名（`sensevoice_engine.cpp:32-33`, `paraformer_engine.cpp:337-340`）
- 还原因：Paraformer 引擎调 `load_tokens_json()`，期待 JSON 数组，但 Release 里是 `<token> <index>` 文本格式

**修复：** 新增 "Stage model files for build" CI 步骤，在 `xcodebuild` 前执行：
```bash
mkdir -p FunASR-iOS/cpp_inference/models/
cp ci_models/sensevoice.int8.onnx   FunASR-iOS/cpp_inference/models/model.int8.onnx
cp ci_models/sensevoice_tokens.txt   FunASR-iOS/cpp_inference/models/tokens.txt
cp ci_models/paraformer_enc.int8.onnx  FunASR-iOS/cpp_inference/models/paraformer_encoder.int8.onnx
cp ci_models/paraformer_dec.int8.onnx  FunASR-iOS/cpp_inference/models/paraformer_decoder.int8.onnx
cp ci_models/paraformer_am.mvn          FunASR-iOS/cpp_inference/models/am.mvn
python3 -c "import json; t=[l.split()[0] for l in open('ci_models/paraformer_tokens.txt') if l.strip()]; open('FunASR-iOS/cpp_inference/models/paraformer_tokens.json','w').write(json.dumps(t,ensure_ascii=False))"
```
同时把 `MODEL_DIR` 改为 `${{ github.workspace }}/FunASR-iOS/cpp_inference/models`

**踩坑：** Python 多行代码写在 YAML `|` 块内时，缩进为 0 的行会终止块，导致 YAML parse error。解决方式：折叠成一行 `python3 -c "..."`。

---

## 问题 3：iOS Simulator 找不到（exit code 70）

**现象：** `Unable to find destination {platform:iOS Simulator, name:iPhone 16, OS=latest}`，可用目标全是 macOS。

**原因：** runner 镜像（macos-15-arm64, 20260224.0170）默认安装 Xcode 16.4，iOS 18.5/18.6 CoreSimulator runtime 由 Xcode 16.4 安装。CI 原来 `xcode-select` 到 Xcode 16.2，其 CoreSimulator 与 iOS 18.5+ runtime 不兼容，导致找不到 iOS 模拟器。

**修复：**
- `Xcode_16.2.app` → `Xcode_16.4.app`
- `OS=latest` → `OS=18.5`（已确认 runner 上存在）

---

## 最终 CI 状态（run 22535875008）

所有基础设施步骤全部通过（✓），最后 "Build and run pipeline tests" 以 **exit code 65** 失败（xcodebuild 测试断言失败，非基础设施问题）。留待下一 session 修复 `ASRPipelineTests.testSenseVoiceSample()` 的实际失败原因。

---

## 涉及文件

- `.github/workflows/ci.yml` — 三次修复（ORT URL / model staging / Xcode 版本）
- `FunASR-iOS/bridge/SenseVoiceContext.mm` — 确认内部文件名（只读）
- `FunASR-iOS/cpp_inference/src/paraformer_engine.cpp` — 确认内部文件名和 JSON token 格式（只读）
- `FunASR-iOS.xcodeproj/project.pbxproj` — 确认 Bundle Resources 内部名（只读）
