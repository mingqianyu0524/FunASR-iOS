# CLAUDE.md — 评测能力搭建 Sprint

## 背景

本分支目标：为 SenseVoice int8 和 Paraformer int8 两个模型搭建评测基础设施，
并将精度评测（CER/SER）整合进 CI 流水线。

**执行环境：** Ubuntu 24.04 LTS（Vultr 2vCPU / 4GB RAM / 80GB SSD）

---

## 本期范围

### ✅ IN SCOPE

| # | 任务 | 说明 |
|---|------|------|
| 1 | 下载数据集 | MagicData-RAMC (OpenSLR #123) + AISHELL-1 (OpenSLR #33) |
| 2 | Python 精度评测 | FunASR Python + ONNX int8 模型，在测试集上跑 CER/SER |
| 3 | 抽取 CI 样本 | RAMC 25 条 + AISHELL-1 25 条，打包上传 GitHub Release |
| 4 | 更新 CI workflow | 新增 fixture 下载、EVAL_AUDIO_DIR 传参、批量测试、SNR 鲁棒曲线 |
| 5 | 更新 Swift 测试代码 | testBatchEval()、CERHelper D/I/S 分解、PR 精度对比输出 |

### ❌ 本期不做（依赖真机或暂无评测结果）

- **RTF / 推理延迟 / 内存测试**：依赖真实 iPhone 硬件，模拟器数字无参考价值
- **流式首包延迟**：依赖真机 + 真实录音链路
- **热降频测试**：依赖真机连续运行
- **Badcase 优化（热词 / 微调）**：尚无正式评测结果，优化方向未定
- **MUSAN 噪声增强评测集**：后续补充

---

## 任务 1 — 下载数据集

> 执行位置：Ubuntu 服务器，使用 /dev/shm（315GB tmpfs）临时存储大文件

```bash
# 创建工作目录
mkdir -p /dev/shm/datasets

# 下载 MagicData-RAMC（约 15GB tar.gz）
# 官方页面：https://www.openslr.org/123/
# 使用 EU 镜像（更快）：
wget -P /dev/shm/datasets \
  https://openslr.elda.org/resources/123/MagicData-RAMC.tar.gz

# 下载 AISHELL-1（约 15GB，含 train/dev/test）
# 官方页面：https://www.openslr.org/33/
wget -P /dev/shm/datasets \
  https://us.openslr.org/resources/33/data_aishell.tgz

# 解压（注意：AISHELL-1 内层是多个 tar 包）
cd /dev/shm/datasets
tar -xzf MagicData-RAMC.tar.gz
tar -xzf data_aishell.tgz
cd data_aishell/wav && for f in *.tar.gz; do tar -xzf "$f"; done
```

解压后结构预期：
```
/dev/shm/datasets/
├── MagicData-RAMC/
│   ├── train/
│   ├── dev/
│   ├── test/              ← 重点：测试集 WAV + 标注
│   └── UTTERANCEINFO.txt
└── data_aishell/
    ├── wav/
    │   ├── train/
    │   ├── dev/
    │   └── test/          ← 重点：测试集 WAV
    └── transcript/
        └── aishell_transcript_v0.8.txt
```

---

## 任务 2 — Python 精度评测

> 目标：获得 SenseVoice int8 和 Paraformer int8 在两个测试集上的 CER/SER 基准数字

### 2.1 安装依赖

```bash
pip install funasr onnxruntime jiwer tqdm
```

### 2.2 下载模型（从 GitHub Release）

```bash
# 从仓库的 models-v1.0 release 下载
mkdir -p ~/asr_models
gh release download models-v1.0 \
  --repo mingqianyu0524/FunASR-iOS \
  --pattern "sensevoice.int8.onnx" \
  --pattern "sensevoice_tokens.txt" \
  --pattern "paraformer_enc.int8.onnx" \
  --pattern "paraformer_dec.int8.onnx" \
  --pattern "paraformer_am.mvn" \
  --pattern "paraformer_tokens.txt" \
  --dir ~/asr_models/
```

### 2.3 评测脚本

在仓库根目录下新建 `scripts/eval_python.py`，实现以下逻辑：

**输入：**
- `--model`：`sensevoice` 或 `paraformer`
- `--model-dir`：模型文件目录路径
- `--audio-dir`：测试集音频目录（递归扫描 `.wav`）
- `--transcript`：对应的参考文本文件（RAMC 的 UTTERANCEINFO.txt 或 AISHELL-1 的 transcript txt）
- `--output`：评测结果 JSON 输出路径

**输出 JSON 格式（供 CI 报告和 PR 对比使用）：**
```json
{
  "model": "sensevoice",
  "dataset": "RAMC-test",
  "total_utterances": 2000,
  "cer": 0.112,
  "ser": 0.341,
  "deletions": 1234,
  "insertions": 567,
  "substitutions": 890,
  "total_ref_chars": 25000,
  "eval_date": "2026-03-03"
}
```

**要求：**
- 使用 `jiwer` 库计算 CER（字符级别，中文去空格）
- SER = 有任意错误的句子数 / 总句子数
- D/I/S 分解通过 `jiwer.compute_measures()` 获取
- 进度条用 `tqdm`
- 脚本末尾打印人类可读的汇总表

**运行示例：**
```bash
# SenseVoice on RAMC test set
python3 scripts/eval_python.py \
  --model sensevoice \
  --model-dir ~/asr_models \
  --audio-dir /dev/shm/datasets/MagicData-RAMC/test \
  --transcript /dev/shm/datasets/MagicData-RAMC/UTTERANCEINFO.txt \
  --output results/sensevoice_ramc.json

# Paraformer on RAMC test set
python3 scripts/eval_python.py \
  --model paraformer \
  --model-dir ~/asr_models \
  --audio-dir /dev/shm/datasets/MagicData-RAMC/test \
  --transcript /dev/shm/datasets/MagicData-RAMC/UTTERANCEINFO.txt \
  --output results/paraformer_ramc.json

# SenseVoice on AISHELL-1 test set
python3 scripts/eval_python.py \
  --model sensevoice \
  --model-dir ~/asr_models \
  --audio-dir /dev/shm/datasets/data_aishell/wav/test \
  --transcript /dev/shm/datasets/data_aishell/transcript/aishell_transcript_v0.8.txt \
  --output results/sensevoice_aishell1.json
```

### 2.4 评测结果归档

将 4 个 JSON 文件（2 模型 × 2 数据集）提交到仓库 `eval_results/` 目录，
作为 baseline 存档。

---

## 任务 3 — 抽取 CI 样本，上传 GitHub Release

> 目标：从测试集中抽取 25+25 条代表性样本，打包上传为 `test-fixtures-v1.0` Release

### 3.1 样本选取原则

**RAMC 25 条（侧重多样性）：**
- 安静室内 ~8 条（UTTERANCEINFO.txt 中 environment=quiet）
- 轻微底噪 ~9 条
- 明显底噪 ~8 条
- 句子时长均匀分布：3-5s、5-10s、10-15s 各占约 1/3
- 男女说话人各半

**AISHELL-1 25 条（随机抽取，确保说话人不重复）：**
- 从 `test/` 目录随机选 25 个不同说话人各 1 条
- 覆盖不同句子长度

### 3.2 样本格式规范

```
ci_fixtures/
├── ramc/
│   ├── ramc_001.wav      # 16kHz 16-bit 单声道 PCM
│   ├── ramc_001.txt      # 参考文本（去标点、去空格，纯汉字）
│   ├── ramc_002.wav
│   ├── ramc_002.txt
│   └── ...（共 25 对）
└── aishell1/
    ├── aishell1_001.wav
    ├── aishell1_001.txt
    └── ...（共 25 对）
```

参考文本格式要求：
- 纯文本，UTF-8
- 去除标点符号
- 去除空格
- 去除数字（保留汉字读法）

### 3.3 打包并上传

```bash
cd ci_fixtures/
tar -czf ramc_fixtures.tar.gz    ramc/
tar -czf aishell1_fixtures.tar.gz aishell1/

# 上传到 GitHub Release（需要 gh auth login 或 GITHUB_TOKEN）
gh release create test-fixtures-v1.0 \
  ramc_fixtures.tar.gz \
  aishell1_fixtures.tar.gz \
  --repo mingqianyu0524/FunASR-iOS \
  --title "CI Test Fixtures v1.0" \
  --notes "25 MagicData-RAMC utterances (academic license) + 25 AISHELL-1 utterances (Apache 2.0). For CI accuracy regression testing only."
```

同时新建 `ci/fixture_versions.txt`，内容为：
```
fixtures-v1.0
```

---

## 任务 4 — 更新 CI workflow

修改 `.github/workflows/ci.yml`，在现有流程基础上新增以下步骤：

### 4.1 新增：下载 CI fixtures

在"Download models"步骤之后插入：

```yaml
# ── CI accuracy fixtures ──────────────────────────────────────
- name: Cache CI fixtures
  id: cache-fixtures
  uses: actions/cache@v4
  with:
    path: ci_fixtures/
    key: ci-fixtures-${{ hashFiles('ci/fixture_versions.txt') }}

- name: Download CI fixtures
  if: steps.cache-fixtures.outputs.cache-hit != 'true'
  env:
    GH_TOKEN: ${{ secrets.GITHUB_TOKEN }}
  run: |
    mkdir -p ci_fixtures
    gh release download test-fixtures-v1.0 \
      --repo ${{ github.repository }} \
      --pattern "ramc_fixtures.tar.gz" \
      --pattern "aishell1_fixtures.tar.gz" \
      --dir ci_fixtures/
    tar -xzf ci_fixtures/ramc_fixtures.tar.gz    -C ci_fixtures/
    tar -xzf ci_fixtures/aishell1_fixtures.tar.gz -C ci_fixtures/
    echo "Fixtures:"
    find ci_fixtures/ -name "*.wav" | wc -l
```

### 4.2 修改：xcodebuild test 命令

在现有 xcodebuild 命令中追加 `EVAL_AUDIO_DIR` 参数：

```yaml
EVAL_AUDIO_DIR=${{ github.workspace }}/ci_fixtures \
```

### 4.3 新增：SNR 鲁棒性曲线 + PR 对比

在"Upload test results"步骤之前插入：

```yaml
- name: Generate accuracy report
  if: always()
  run: |
    # 从 build.log 中提取测试输出并生成 Markdown 表格
    python3 scripts/parse_test_results.py \
      --log build.log \
      --baseline-dir eval_results/ \
      --output accuracy_report.md
    cat accuracy_report.md >> $GITHUB_STEP_SUMMARY
```

新建 `scripts/parse_test_results.py`，功能：
- 从 `build.log` 提取 `testBatchEval` 输出的 CER/SER 数字
- 与 `eval_results/` 下的 baseline JSON 对比，计算 delta
- 生成 Markdown 表格，写入 `$GITHUB_STEP_SUMMARY`（显示在 Actions run 页面）
- 如果 CER 相比 baseline 劣化 > 5%，以非零状态退出（可选，用于 CI 阻断）

**SNR 曲线**（暂以 GitHub Step Summary Markdown 表格形式呈现，不生成图片）：
- 测试代码在不同 SNR 条件下运行时（见任务 5），输出各 SNR 档位的 CER
- `parse_test_results.py` 提取并格式化为表格

### 4.4 更新 xcscheme

在 `FunASR-iOS.xcodeproj/xcshareddata/xcschemes/FunASR-iOS.xcscheme` 的
`<EnvironmentVariables>` 节点追加：

```xml
<EnvironmentVariable
  key   = "EVAL_AUDIO_DIR"
  value = "$(EVAL_AUDIO_DIR)"
  isEnabled = "YES">
</EnvironmentVariable>
```

---

## 任务 5 — 更新 Swift 测试代码

### 5.1 CERHelper.swift — 新增 D/I/S 分解

在现有 `cer()` 方法基础上，新增返回 `(cer: Double, d: Int, i: Int, s: Int)` 的
`cerWithDetails()` 方法，通过回溯 DP 矩阵获取操作类型统计。

### 5.2 ASRPipelineTests.swift — 新增 testBatchEval()

新增测试方法，逻辑如下：
- 读取环境变量 `EVAL_AUDIO_DIR`，若不存在则跳过（`throw XCTSkip`）
- 扫描 `ramc/` 和 `aishell1/` 两个子目录，逐一加载 .wav + .txt 对
- 对每条音频：SenseVoice 推理 → 计算 CER/D/I/S → 累积统计
- 分组打印汇总结果（格式见下）：

```
── Batch Eval: RAMC (25 utterances) ──
CER:   12.3%   SER:  45.2%
D:  234  I:  67  S: 189  Ref chars: 4012
── Batch Eval: AISHELL-1 (25 utterances) ──
CER:    8.7%   SER:  32.0%
D:  102  I:  41  S: 118  Ref chars: 3251
```

- 断言：两组 CER 均 < 30%（宽松阈值，防止明显回归，不作为绝对精度门槛）

### 5.3 SNR 鲁棒性测试（选做，如时间允许）

在 `testBatchEval()` 中对 AISHELL-1 样本额外做噪声叠加测试：
- 用 Swift 生成高斯白噪声，以 5dB / 10dB / 20dB SNR 叠加到原始 PCM
- 分别推理并记录 CER
- 打印 SNR 曲线表（供 CI Step Summary 解析）

---

## 文件变更清单

| 文件 | 操作 | 说明 |
|------|------|------|
| `CLAUDE.md` | 新建（本文件） | 任务规范 |
| `ci/fixture_versions.txt` | 新建 | 内容：`fixtures-v1.0` |
| `eval_results/sensevoice_ramc.json` | 新建 | 评测 baseline（任务 2 产出） |
| `eval_results/sensevoice_aishell1.json` | 新建 | 评测 baseline |
| `eval_results/paraformer_ramc.json` | 新建 | 评测 baseline |
| `eval_results/paraformer_aishell1.json` | 新建 | 评测 baseline |
| `scripts/eval_python.py` | 新建 | Python 评测脚本 |
| `scripts/prepare_fixtures.sh` | 新建 | 样本抽取 + 打包脚本 |
| `scripts/parse_test_results.py` | 新建 | CI 报告生成脚本 |
| `.github/workflows/ci.yml` | 修改 | 新增 fixtures 步骤 + accuracy report |
| `FunASR-iOS.xcodeproj/.../FunASR-iOS.xcscheme` | 修改 | 新增 EVAL_AUDIO_DIR 环境变量 |
| `FunASR-iOSTests/CERHelper.swift` | 修改 | 新增 cerWithDetails() |
| `FunASR-iOSTests/ASRPipelineTests.swift` | 修改 | 新增 testBatchEval() |

---

## 验收标准

1. `scripts/eval_python.py` 在 RAMC test set 上能输出 SenseVoice 和 Paraformer 的 CER/SER
2. `eval_results/` 下有 4 个 baseline JSON 文件已提交
3. GitHub Release `test-fixtures-v1.0` 存在，包含两个 tar.gz，合计 < 30MB
4. CI 运行（GitHub Actions）能正确下载 fixtures，`testBatchEval` 通过
5. CI Summary 页面有精度对比表（与 baseline 的 delta）
