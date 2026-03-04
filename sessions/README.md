# Claude Session Logs

每次 Claude 任务结束后，小结以 `<commit_hash>_<one-line-summary>.md` 格式存放在此目录。

## 查看方式

```bash
# 列出所有 session
git log --oneline claude-session-logs

# 查看某个 session
git show claude-session-logs:sessions/<filename>.md

# 或者切换到这个分支直接读
git checkout claude-session-logs
ls sessions/
```

## 日志索引

| 文件 | 摘要 |
|------|------|
| `dbb14ae4_ci-pipeline-fix-onnx-upload-and-model-staging.md` | 修复 CI：ORT URL / 模型文件名映射 / Xcode 16.4 / exit 65 |
| `65e00c5_ci-green-test-fixture-and-special-token-stripping.md` | CI 全绿：加入测试音频 fixture + 去除 SenseVoice 特殊 token |
| `363b8d5_eval-infrastructure-batch-cer-ci-fixtures.md` | 评测能力搭建：Python eval 脚本 / 批量 CER / CI fixture 流水线 |
