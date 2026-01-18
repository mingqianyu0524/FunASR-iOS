#pragma once

#include <string>
#include <map>
#include <vector>

class WhisperTokenizer {
public:
    WhisperTokenizer();
    /**
     * @brief 加载词汇表 (Load Vocabulary)
     * * 从指定的 JSON 文件中读取 token 到 string 的映射关系。
     * * @param vocab_path 词汇表文件的路径 (例如: "./vocab.json")
     * @return bool 如果加载成功返回 true，文件不存在或解析失败返回 false
     */
    bool load(const std::string& vocab_path);

    /**
     * @brief 解码单个 Token (Decode Single Token)
     * * 将一个整数 Token ID 转换为其对应的字符串文本。
     * * @param token_id 模型输出的整数 ID (例如: 220)
     * @return std::string 对应的文本片段 (例如: " ")。
     * 如果 ID 不在词汇表中，建议返回 "<UNK>" 或空字符串。
     */
    std::string decode(int token_id) const;

    /**
     * @brief (可选) 解码一串 Tokens (Decode Batch)
     * * 这是一个辅助函数，用于一次性将整个 ID 序列转换成完整的句子。
     * * @param token_ids 整数 ID 的列表 (例如: {50258, 220, 1234})
     * @return std::string 拼接好的完整字符串
     */
    std::string decode(const std::vector<int>& token_ids) const;

    // ----- 特殊 Token 判断辅助函数 (可选) -----

    /**
     * @brief 检查是否是结束标记 (Is End of Text?)
     * 用于判断推理是否应该停止。
     * 通常 Whisper 的 EOT token id 是 50257
     */
    bool is_eot(int token_id) const;
private:
    // 核心数据结构: 存储从 int ID 到 string 文本的映射
    std::map<int, std::string> vocab_;

    // 特殊 Token ID 常量 (你可以根据 dump_tokenizer.py 的输出硬编码这些值)
    const int SOT_TOKEN_ID = 50258; // <|startoftranscript|>
    const int EOT_TOKEN_ID = 50257; // <|endoftext|>
};