#pragma once

#include <string>
#include <vector>
#include <memory>
#include "onnxruntime/onnxruntime_cxx_api.h"
#include "sensevoice_tokenizer.h"
#include "feature_extractor.h"

class SenseVoiceEngine {
public:
    SenseVoiceEngine();
    ~SenseVoiceEngine();

    /**
     * @brief Initialize engine: load model and vocabulary
     * @param model_path  SenseVoice ONNX model path (int8)
     * @param tokens_path Token ID to string mapping file
     * @return bool whether initialization succeeded
     */
    bool initialize(const std::string& model_path,
                    const std::string& tokens_path);

    /**
     * @brief Run speech-to-text inference (single forward pass + CTC decode)
     * @param pcm_data 16kHz mono audio data
     * @return Recognized text string
     */
    std::string transcribe(const std::vector<float>& pcm_data);

private:
    // ONNX Runtime
    Ort::Env env_;
    Ort::SessionOptions session_options_;
    std::unique_ptr<Ort::Session> session_;

    // Components
    SenseVoiceTokenizer tokenizer_;
    FeatureExtractor feature_extractor_;

    // CMVN parameters (loaded from model metadata)
    std::vector<float> neg_mean_;   // 560 values
    std::vector<float> inv_stddev_; // 560 values

    // SenseVoice special IDs (from model metadata)
    int lang_zh_id_ = 3;
    int with_itn_id_ = 14;
    int blank_id_ = 0;
    int vocab_size_ = 25055;

    // LFR parameters (from model metadata)
    int lfr_window_size_ = 7;
    int lfr_window_shift_ = 6;

    // Helpers
    bool load_metadata();
    std::vector<float> parse_csv(const std::string& csv);

    // Feature processing
    std::vector<float> apply_lfr(const std::vector<float>& fbank, int n_frames,
                                  int n_mels, int& out_frames);
    void apply_cmvn(std::vector<float>& features, int n_frames, int feat_dim);

    // CTC greedy decode
    std::string ctc_greedy_decode(const float* logits, int T, int vocab_size);
};
