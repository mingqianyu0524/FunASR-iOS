#pragma once

#include <string>
#include <vector>
#include <memory>
#include "onnxruntime/onnxruntime_cxx_api.h"
#include "whisper_tokenizer.h"
#include "feature_extractor.h"

class WhisperEngine {
public:
    WhisperEngine();
    ~WhisperEngine();

    /**
     * @brief 初始化引擎：加载模型和词汇表
     * @param encoder_path Encoder ONNX 模型路径
     * @param decoder_path Decoder ONNX 模型路径
     * @param vocab_path   Tokenizer 词汇表路径
     * @return bool 是否初始化成功
     */
    bool initialize(const std::string& encoder_path, 
                    const std::string& decoder_path, 
                    const std::string& vocab_path);

    /**
     * @brief 执行语音转文字推理
     * @param pcm_data 16kHz 单声道音频数据 (通常为 30秒 / 480k 采样点)
     * @return 识别出的文本字符串
     */
    std::string transcribe(const std::vector<float>& pcm_data);

private:
    // ONNX Runtime 环境
    Ort::Env env_;
    Ort::SessionOptions session_options_;
    std::unique_ptr<Ort::Session> encoder_session_;
    std::unique_ptr<Ort::Session> decoder_session_;
    
    // 组件
    WhisperTokenizer tokenizer_;
    FeatureExtractor feature_extractor_;

    // 常量定义
    static constexpr int64_t INPUT_BATCH = 1;
    static constexpr int64_t N_MELS = 80;
    static constexpr int64_t N_FRAMES = 3000;
    
    static constexpr int SOT_TOKEN = 50258; // <|startoftranscript|>
    static constexpr int EOT_TOKEN = 50257; // <|endoftext|>
};
