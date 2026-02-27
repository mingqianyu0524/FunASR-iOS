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
     * @brief Initialize engine: load models and vocabulary
     * @param encoder_path Encoder ONNX model path (fp16)
     * @param decoder_path Decoder ONNX model path (fp16, with embedded token embedding layer)
     * @param vocab_path   Tokenizer vocabulary path
     * @param embedder_path   (unused, kept for API compatibility)
     * @return bool whether initialization succeeded
     */
    bool initialize(const std::string& encoder_path,
                    const std::string& decoder_path,
                    const std::string& vocab_path,
                    const std::string& embedder_path);

    /**
     * @brief Run speech-to-text inference
     * @param pcm_data 16kHz mono audio data
     * @return Recognized text string
     */
    std::string transcribe(const std::vector<float>& pcm_data);

private:
    // ONNX Runtime environment
    Ort::Env env_;
    Ort::SessionOptions session_options_;
    std::unique_ptr<Ort::Session> encoder_session_;
    std::unique_ptr<Ort::Session> decoder_session_;

    // Components
    WhisperTokenizer tokenizer_;
    FeatureExtractor feature_extractor_;

    // Constants
    static constexpr int64_t INPUT_BATCH = 1;
    static constexpr int64_t N_MELS = 80;
    static constexpr int64_t N_FRAMES = 3000;

    static constexpr int SOT_TOKEN = 50258; // <|startoftranscript|>
    static constexpr int EOT_TOKEN = 50257; // <|endoftext|>
};
