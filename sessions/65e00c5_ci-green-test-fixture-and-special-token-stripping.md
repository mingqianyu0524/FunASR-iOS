# Session 65e00c5 — CI 全绿：补充测试音频 fixture + 去除特殊 token

**日期：** 2026-03-01
**分支：** claude/pipeline-setup-isL66
**结果：** CI run 22536272974 全部步骤通过（绿色）✓

---

## 问题根因

上个 session 遗留的 `ASRPipelineTests.testSenseVoiceSample()` 失败（0.630 秒内失败），两个根因：

### 根因 1：测试音频不存在

`ramc_sample.wav` 从未提交到仓库，测试在 `XCTUnwrap(bundle.url(forResource: "ramc_sample", withExtension: "wav"))` 处立即抛出。

**关键细节：** `FunASR-iOSTests` 目标使用 `PBXFileSystemSynchronizedRootGroup`（Xcode 16 新特性），`FunASR-iOSTests/` 目录下的所有文件会自动包含在 test bundle 中，无需手动在 `.xcodeproj` 里添加引用。`.gitignore` 已有 `!FunASR-iOSTests/fixtures/*.wav` 例外规则，允许提交 WAV。

**修复：** 把 FunASR 模型缓存中自带的 `asr_example.wav`（5.55 秒，16kHz 单声道 PCM，174 KB）复制为 `FunASR-iOSTests/fixtures/ramc_sample.wav` 并提交。

### 根因 2：参考文本是占位符

`ramc_sample_ref.txt` 内容为 `PLACEHOLDER: replace with actual reference transcription`。

**修复：**
- 用 FunASR Python 模型（SenseVoice + Paraformer）对音频做推理，确认内容为：`欢迎大家来体验达摩院推出的语音识别模型`
- 更新 `ramc_sample_ref.txt` 为该文本

### 根因 3：CER 计算前未剥离特殊 token

SenseVoice 的 CTC 输出包含语言 / 情绪 / 事件标签：`<|zh|><|NEUTRAL|><|Speech|><|withitn|>`

若 hypothesis 含这些 tag，reference 为纯中文，字符编辑距离 ≈ 39（tag 总字符数），而 reference 仅 19 字，CER ≈ 205%，远超 20% 阈值。

**修复：** 在 `ASRPipelineTests.swift` 中新增 `stripSpecialTokens()` helper，用 `NSRegularExpression(pattern: "<\\|[^|>]*\\|>")` 在 CER 计算前去除所有 `<|...|>` 标签。

---

## 修改文件

| 文件 | 变更 |
|------|------|
| `FunASR-iOSTests/fixtures/ramc_sample.wav` | 新增，174 KB，FunASR 标准示例音频 |
| `FunASR-iOSTests/fixtures/ramc_sample_ref.txt` | `PLACEHOLDER` → `欢迎大家来体验达摩院推出的语音识别模型` |
| `FunASR-iOSTests/ASRPipelineTests.swift` | 在 `let text = ...` 处调用 `stripSpecialTokens()`；新增该 helper（8 行）|

---

## CI 验证

Run 22536272974 (`workflow_dispatch` on `claude/pipeline-setup-isL66`)：

```
✓ Set up job
✓ Select Xcode 16.4
✓ Check iOS Simulator Runtime
✓ List available simulators
✓ Cache ONNX Runtime
✓ Download ONNX Runtime
✓ Cache ASR models
✓ Download models from GitHub Releases
✓ Stage model files for build
✓ Build and run pipeline tests        ← 之前一直在这里失败
✓ Upload test results
```
