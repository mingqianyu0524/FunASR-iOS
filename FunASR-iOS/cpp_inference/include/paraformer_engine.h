#pragma once

#include <string>
#include <vector>
#include <memory>
#include "onnxruntime/onnxruntime_cxx_api.h"
#include "feature_extractor.h"
#include "inference_metrics.h"

/**
 * ParaformerEngine — Non-autoregressive ASR with CIF predictor + parallel decoder
 *
 * Two ONNX models:
 *   Encoder: speech[1,T_lfr,560] + speech_lengths[1]
 *            -> enc[1,T,512] + enc_len[1] + alphas[1,T]
 *   Decoder: enc + enc_len + acoustic_embeds[1,N,512] + acoustic_embeds_len[1]
 *            + 16x in_cache[1,512,10]
 *            -> logits[1,N,8404] + sample_ids[1,N] + 16x out_cache
 *
 * Pipeline: PCM -> Fbank(80) -> LFR(7,6) -> CMVN(560) -> Encoder
 *           -> CIF(alphas, enc) -> acoustic_embeds -> Decoder -> detokenize
 */
class ParaformerEngine {
public:
    ParaformerEngine();
    ~ParaformerEngine();

    bool initialize(const std::string& model_dir);

    std::string transcribe(const std::vector<float>& pcm_data);
    std::string transcribe(const std::vector<float>& pcm_data, InferenceMetrics& metrics);

    // Streaming interface
    void start_stream();
    std::string feed_chunk(const std::vector<float>& pcm_chunk);
    std::string end_stream(InferenceMetrics& metrics);
    bool supports_streaming() const { return true; }

private:
    // ONNX Runtime — two sessions
    Ort::Env env_;
    Ort::SessionOptions session_options_;
    std::unique_ptr<Ort::Session> encoder_session_;
    std::unique_ptr<Ort::Session> decoder_session_;
    bool initialized_ = false;

    // Feature extraction (80-dim fbank, shared with SenseVoice)
    FeatureExtractor feature_extractor_;

    // LFR parameters (same as SenseVoice)
    static constexpr int LFR_WINDOW_SIZE = 7;
    static constexpr int LFR_WINDOW_SHIFT = 6;
    static constexpr int LFR_FEAT_DIM = 560;  // 7 * 80

    // CMVN parameters (560-dim, applied after LFR)
    std::vector<float> cmvn_neg_mean_;  // 560 values
    std::vector<float> cmvn_istd_;      // 560 values

    // Tokenizer
    std::vector<std::string> vocab_;
    int vocab_size_ = 0;
    static constexpr int BLANK_ID = 0;
    static constexpr int EOS_ID = 2;

    // Decoder cache constants
    static constexpr int NUM_DECODER_LAYERS = 16;
    static constexpr int DECODER_HIDDEN_DIM = 512;
    static constexpr int DECODER_CACHE_TIME = 10;
    static constexpr int ENCODER_OUTPUT_DIM = 512;

    // Streaming state
    bool stream_active_ = false;
    std::vector<float> stream_pcm_buffer_;
    std::vector<std::vector<float>> decoder_caches_;  // 16 x [512 * 10]

    // Internal methods
    bool load_cmvn(const std::string& mvn_path);
    bool load_tokens_json(const std::string& tokens_path);

    // LFR: [n_frames, 80] -> [out_frames, 560]
    std::vector<float> apply_lfr(const std::vector<float>& fbank,
                                  int n_frames, int n_mels, int& out_frames);

    // CMVN: (x + neg_mean) * istd on 560-dim LFR features
    void apply_cmvn(std::vector<float>& features, int n_frames, int feat_dim);

    // CIF predictor: accumulate alphas, fire when >= 1.0
    struct CIFResult {
        std::vector<float> acoustic_embeds;  // [n_tokens * hidden_dim]
        int n_tokens = 0;
    };
    CIFResult run_cif(const float* encoder_out, const float* alphas,
                       int T, int hidden_dim);

    // Run encoder ONNX session
    struct EncoderOutput {
        std::vector<float> enc;      // [1, T, 512]
        int enc_len = 0;
        std::vector<float> alphas;   // [T]
        int T = 0;
    };
    EncoderOutput run_encoder(const std::vector<float>& lfr_features, int n_lfr_frames);

    // Run decoder ONNX session
    std::vector<int64_t> run_decoder(const EncoderOutput& enc_out,
                                      const CIFResult& cif_result);

    // Token IDs -> string
    std::string detokenize(const std::vector<int64_t>& token_ids);

    // Initialize decoder caches to zeros
    void reset_decoder_caches();
};
