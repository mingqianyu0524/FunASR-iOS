# Session 363b8d5 — 评测能力搭建：批量 CER / CI fixture / Python 评测脚本

**日期：** 2026-03-04
**分支：** claude/test-isL66
**任务：** 拉取 claude/test-isL66 分支，按 CLAUDE.md（评测 Sprint）完成 Phase 4 评测基础设施搭建

---

## 背景

本 session 对应 CLAUDE.md 的"评测能力搭建 Sprint"，目标是为 SenseVoice int8 和 Paraformer int8 搭建完整的精度评测基础设施，并将批量 CER 测试整合进 CI 流水线。

CLAUDE.md 定义了 5 个任务：

| 任务 | 说明 |
|------|------|
| 1 | 下载数据集（RAMC + AISHELL-1，需手动在服务器执行） |
| 2 | Python 精度评测脚本（CER/SER/D/I/S） |
| 3 | 抽取 CI fixture 并上传 GitHub Release |
| 4 | 更新 CI workflow（fixture 下载 + accuracy report） |
| 5 | 更新 Swift 测试代码（testBatchEval + CERHelper D/I/S） |

本 session 完成了任务 2、4、5 的全部代码，以及任务 3 的脚本（`prepare_fixtures.sh`）；任务 1/3 的数据集下载和 GitHub Release 上传需在服务器手动执行。

---

## 变更详情

### 新增文件

#### `scripts/eval_python.py`（598 行）

端到端 Python 评测脚本，支持：
- `--model sensevoice` / `paraformer`
- `--audio-dir`：递归扫描 `.wav`
- `--transcript`：自动检测 RAMC (`UTTERANCEINFO.txt` tab 格式) / AISHELL-1 (`aishell_transcript_v0.8.txt` 空格格式)
- 内置 Kaldi-compatible 80-dim log-mel fbank + LFR（SenseVoice）/ CMVN（Paraformer）
- CIF forward pass（简化版，用于 Paraformer offline decode）
- 输出 JSON：`{model, dataset, total_utterances, cer, ser, deletions, insertions, substitutions, total_ref_chars, eval_date}`
- 打印人类可读汇总表

**运行示例：**
```bash
python3 scripts/eval_python.py \
  --model sensevoice \
  --model-dir ~/asr_models \
  --audio-dir /dev/shm/datasets/MagicData-RAMC/test \
  --transcript /dev/shm/datasets/MagicData-RAMC/UTTERANCEINFO.txt \
  --output eval_results/sensevoice_ramc.json
```

#### `scripts/prepare_fixtures.sh`（320 行）

分层抽样脚本：
- RAMC 25 条：按 `quiet/light/heavy` 环境类型 (8/9/8) 分层，剔除 < 5 或 > 60 汉字的句子
- AISHELL-1 25 条：每个说话人最多 1 条，覆盖 25 个不同说话人
- 自动用 `ffmpeg` 标准化为 16kHz 16-bit mono PCM WAV
- 打包为 `ramc_fixtures.tar.gz` / `aishell1_fixtures.tar.gz`
- 打印 `gh release create test-fixtures-v1.0` 命令供手动执行

#### `scripts/parse_test_results.py`（214 行）

CI 报告生成脚本：
- 从 `build.log` 用正则提取 `testBatchEval()` 的结构化输出（CER/SER/D/I/S）
- 与 `eval_results/` baseline JSON 对比，计算 delta（pp，百分点）
- 输出 Markdown 表格 + 趋势箭头（⬆⬇→）
- 写入 `$GITHUB_STEP_SUMMARY`（Actions run 页面可见）
- `--fail-on-regression`：CER 劣化 > 5pp 时非零退出（可选 CI 阻断）

#### `ci/fixture_versions.txt`

内容：`fixtures-v1.0`，用于 `actions/cache` 的 cache key。

#### `eval_results/*.json`（4 个占位文件）

`sensevoice_ramc.json`、`sensevoice_aishell1.json`、`paraformer_ramc.json`、`paraformer_aishell1.json`，`cer`/`ser` 等字段为 `null`，待 full eval run 后填入真实数字。

---

### 修改文件

#### `FunASR-iOSTests/CERHelper.swift`

新增 `cerWithDetails()` 方法：
- 完整 2D Levenshtein DP（每格存 `(dist, del, ins, sub)` 四元组）
- 通过比较 substitution/deletion/insertion 代价选最优操作并累积计数
- 原 `cer()` 改为调用 `cerWithDetails().cer`（行为不变）

```swift
static func cerWithDetails(hypothesis: String, reference: String)
    -> (cer: Double, deletions: Int, insertions: Int, substitutions: Int)
```

#### `FunASR-iOSTests/ASRPipelineTests.swift`

新增 `testBatchEval()` 测试方法：
- 读取 `EVAL_AUDIO_DIR` 环境变量；未设置时 `throw XCTSkip`（不阻断 smoke test）
- 扫描 `EVAL_AUDIO_DIR/ramc/` 和 `aishell1/` 两个子目录（`.wav` + 同名 `.txt`）
- 对每条音频：`SenseVoiceContext.transcribeData(withMetrics:)` → `stripSpecialTokens` → `cerWithDetails()` → 累积 D/I/S
- 按固定格式打印（供 `parse_test_results.py` 正则提取）：
  ```
  ── Batch Eval: RAMC (25 utterances) ──
  CER:   12.3%   SER:  45.2%
  D:  234  I:  67  S: 189  Ref chars: 4012
  ```
- `XCTAssertLessThan(cer, 0.30)` 宽松防回归断言（两个数据集各一条）

新增 `addNoise(to:snrDB:)` 辅助方法：LCG + Box-Muller 生成高斯白噪声，用于 SNR 鲁棒性测试。

新增 `loadWAVFromURL()` 辅助方法：支持从任意 URL 加载 WAV（批量扫描目录时使用，不依赖 bundle）。

#### `.github/workflows/ci.yml`

在 model 下载步骤之后，`xcodebuild test` 之前插入：

```yaml
- name: Cache CI fixtures
  id: cache-fixtures
  uses: actions/cache@v4
  with:
    path: ci_fixtures/
    key: ci-fixtures-${{ hashFiles('ci/fixture_versions.txt') }}

- name: Download CI fixtures
  if: steps.cache-fixtures.outputs.cache-hit != 'true'
  ...
  # gh release download test-fixtures-v1.0 + tar 解压
```

在 `xcodebuild test` 命令中追加 `EVAL_AUDIO_DIR=${{ github.workspace }}/ci_fixtures`。

在"Upload test results"前插入：

```yaml
- name: Generate accuracy report
  if: always()
  run: |
    python3 scripts/parse_test_results.py \
      --log build.log \
      --baseline-dir eval_results/ \
      --output accuracy_report.md
    cat accuracy_report.md >> "$GITHUB_STEP_SUMMARY"
```

`accuracy_report.md` 也加入 artifact 上传。

#### `FunASR-iOS.xcodeproj/xcshareddata/xcschemes/FunASR-iOS.xcscheme`

在 `<EnvironmentVariables>` 节点追加：

```xml
<EnvironmentVariable
   key = "EVAL_AUDIO_DIR"
   value = "$(EVAL_AUDIO_DIR)"
   isEnabled = "YES">
</EnvironmentVariable>
```

---

## 待手动完成（需服务器操作）

| 步骤 | 命令 / 说明 |
|------|-------------|
| 下载 RAMC | `wget https://openslr.elda.org/resources/123/MagicData-RAMC.tar.gz` |
| 下载 AISHELL-1 | `wget https://us.openslr.org/resources/33/data_aishell.tgz` |
| 跑全量 eval | `python3 scripts/eval_python.py --model sensevoice ...` × 4 |
| 更新 baseline | 把 eval 结果覆写 `eval_results/*.json` 并提交 |
| 生成 fixtures | `bash scripts/prepare_fixtures.sh --ramc-dir ... --aishell-dir ...` |
| 上传 Release | `gh release create test-fixtures-v1.0 ramc_fixtures.tar.gz aishell1_fixtures.tar.gz` |
| 推送分支 | `git push origin claude/test-isL66` → CI 自动跑 `testBatchEval()` |

---

## 验收状态

| 验收标准 | 状态 |
|---------|------|
| `scripts/eval_python.py` 可在 RAMC test set 输出 CER/SER | ✅ 代码完成，待在服务器执行 |
| `eval_results/` 有 4 个 baseline JSON | ⚠️ 占位（`cer: null`），待 full eval 填入 |
| GitHub Release `test-fixtures-v1.0` 存在 | ⏳ 待手动上传 |
| CI `testBatchEval` 通过 | ⏳ 待 fixtures Release 就绪后验证 |
| CI Summary 有精度对比表 | ⏳ 同上 |
