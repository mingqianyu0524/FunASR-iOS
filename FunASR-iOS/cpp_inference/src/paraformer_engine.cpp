#include "paraformer_engine.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cmath>
#include <numeric>

// =====================================================================
// ParaformerEngine — Non-autoregressive ASR with CIF + parallel decoder
//
// Two ONNX models (encoder + decoder):
//   Encoder: speech[1,T_lfr,560] + speech_lengths[1]
//            -> enc[1,T,512] + enc_len[1] + alphas[1,T]
//   Decoder: enc + enc_len + acoustic_embeds[1,N,512] + acoustic_embeds_len[1]
//            + 16x in_cache[1,512,10]
//            -> logits[1,N,8404] + sample_ids[1,N] + 16x out_cache
//
// Pipeline: PCM -> Fbank(80) -> LFR(7,6) -> CMVN(560) -> Encoder
//           -> CIF(alphas,enc) -> acoustic_embeds -> Decoder -> detokenize
// =====================================================================

ParaformerEngine::ParaformerEngine()
    : env_(ORT_LOGGING_LEVEL_WARNING, "Paraformer")
{
}

ParaformerEngine::~ParaformerEngine() = default;

// =====================================================================
// Load CMVN from am.mvn (Kaldi format, 560-dim after LFR)
// Format:
//   <AddShift> 560 560
//   <LearnRateCoef> 0 [ val1 val2 ... val560 ]
//   <Rescale> 560 560
//   <LearnRateCoef> 0 [ val1 val2 ... val560 ]
// =====================================================================
bool ParaformerEngine::load_cmvn(const std::string& mvn_path) {
    std::ifstream file(mvn_path);
    if (!file.is_open()) {
        std::cerr << "Failed to open CMVN file: " << mvn_path << std::endl;
        return false;
    }

    cmvn_neg_mean_.clear();
    cmvn_istd_.clear();

    std::string line;
    enum State { SEEK_ADDSHIFT, READ_MEAN, SEEK_RESCALE, READ_ISTD, DONE };
    State state = SEEK_ADDSHIFT;

    while (std::getline(file, line) && state != DONE) {
        auto pos = line.find_first_not_of(" \t");
        if (pos == std::string::npos) continue;
        line = line.substr(pos);

        switch (state) {
            case SEEK_ADDSHIFT:
                if (line.find("<AddShift>") != std::string::npos) {
                    state = READ_MEAN;
                }
                break;
            case READ_MEAN: {
                std::stringstream ss(line);
                std::string token;
                while (ss >> token) {
                    if (token == "[" || token == "]") continue;
                    try {
                        cmvn_neg_mean_.push_back(std::stof(token));
                    } catch (...) {}
                }
                if (line.find(']') != std::string::npos) {
                    state = SEEK_RESCALE;
                }
                break;
            }
            case SEEK_RESCALE:
                if (line.find("<Rescale>") != std::string::npos) {
                    state = READ_ISTD;
                }
                break;
            case READ_ISTD: {
                std::stringstream ss(line);
                std::string token;
                while (ss >> token) {
                    if (token == "[" || token == "]") continue;
                    try {
                        cmvn_istd_.push_back(std::stof(token));
                    } catch (...) {}
                }
                if (line.find(']') != std::string::npos) {
                    state = DONE;
                }
                break;
            }
            default:
                break;
        }
    }

    // Kaldi am.mvn may have an extra trailing count/bias value per row — truncate
    if ((int)cmvn_neg_mean_.size() == LFR_FEAT_DIM + 1) {
        cmvn_neg_mean_.pop_back();
    }
    if ((int)cmvn_istd_.size() == LFR_FEAT_DIM + 1) {
        cmvn_istd_.pop_back();
    }

    if ((int)cmvn_neg_mean_.size() != LFR_FEAT_DIM ||
        (int)cmvn_istd_.size() != LFR_FEAT_DIM) {
        std::cerr << "CMVN dimension mismatch: neg_mean=" << cmvn_neg_mean_.size()
                  << " istd=" << cmvn_istd_.size()
                  << " (expected " << LFR_FEAT_DIM << ")" << std::endl;
        return false;
    }

    std::cout << "  CMVN loaded (" << LFR_FEAT_DIM << "-dim) from " << mvn_path << std::endl;
    return true;
}

// =====================================================================
// Load tokens from JSON array: ["<blank>", "<s>", "</s>", "token3", ...]
// Index in array = token ID
// =====================================================================
bool ParaformerEngine::load_tokens_json(const std::string& tokens_path) {
    std::ifstream file(tokens_path);
    if (!file.is_open()) {
        std::cerr << "Failed to open tokens file: " << tokens_path << std::endl;
        return false;
    }

    // Simple JSON array parser (no external dependency needed)
    std::string content((std::istreambuf_iterator<char>(file)),
                         std::istreambuf_iterator<char>());

    vocab_.clear();
    bool in_string = false;
    bool escape = false;
    std::string current_token;

    for (char c : content) {
        if (escape) {
            current_token += c;
            escape = false;
            continue;
        }
        if (c == '\\' && in_string) {
            escape = true;
            continue;
        }
        if (c == '"') {
            if (in_string) {
                // End of string — emit token
                vocab_.push_back(current_token);
                current_token.clear();
            }
            in_string = !in_string;
            continue;
        }
        if (in_string) {
            current_token += c;
        }
    }

    vocab_size_ = static_cast<int>(vocab_.size());
    std::cout << "  Loaded " << vocab_size_ << " tokens from JSON" << std::endl;
    return vocab_size_ > 0;
}

// =====================================================================
// LFR: stack LFR_WINDOW_SIZE frames, subsample by LFR_WINDOW_SHIFT
//   [n_frames, 80] -> [out_frames, 560]
// =====================================================================
std::vector<float> ParaformerEngine::apply_lfr(const std::vector<float>& fbank,
                                                int n_frames, int n_mels,
                                                int& out_frames)
{
    const int W = LFR_WINDOW_SIZE;  // 7
    const int S = LFR_WINDOW_SHIFT; // 6
    const int feat_dim = W * n_mels; // 560

    out_frames = (n_frames - W) / S + 1;
    if (out_frames <= 0) {
        out_frames = 1;
        std::vector<float> result(feat_dim, 0.0f);
        for (int j = 0; j < std::min(n_frames, W); ++j) {
            std::copy(fbank.begin() + j * n_mels,
                      fbank.begin() + j * n_mels + n_mels,
                      result.begin() + j * n_mels);
        }
        return result;
    }

    std::vector<float> result(out_frames * feat_dim, 0.0f);

    for (int i = 0; i < out_frames; ++i) {
        int start_frame = i * S;
        for (int j = 0; j < W; ++j) {
            int src_frame = start_frame + j;
            if (src_frame < n_frames) {
                std::copy(fbank.begin() + src_frame * n_mels,
                          fbank.begin() + src_frame * n_mels + n_mels,
                          result.begin() + i * feat_dim + j * n_mels);
            }
        }
    }
    return result;
}

// =====================================================================
// CMVN: (x + neg_mean) * istd — applied on 560-dim LFR features
// =====================================================================
void ParaformerEngine::apply_cmvn(std::vector<float>& features,
                                   int n_frames, int feat_dim) {
    if ((int)cmvn_neg_mean_.size() != feat_dim ||
        (int)cmvn_istd_.size() != feat_dim) {
        throw std::runtime_error("Paraformer CMVN dimension mismatch");
    }

    for (int i = 0; i < n_frames; ++i) {
        float* frame = features.data() + i * feat_dim;
        for (int j = 0; j < feat_dim; ++j) {
            frame[j] = (frame[j] + cmvn_neg_mean_[j]) * cmvn_istd_[j];
        }
    }
}

// =====================================================================
// CIF Predictor
//   Accumulate alpha weights per encoder timestep.
//   When accumulated >= 1.0, fire: emit weighted acoustic embedding.
// =====================================================================
ParaformerEngine::CIFResult ParaformerEngine::run_cif(
    const float* encoder_out, const float* alphas,
    int T, int hidden_dim)
{
    CIFResult result;

    std::vector<float> current_embed(hidden_dim, 0.0f);
    float accumulated = 0.0f;
    constexpr float threshold = 1.0f;

    for (int t = 0; t < T; ++t) {
        float alpha = alphas[t];
        accumulated += alpha;
        const float* enc_frame = encoder_out + t * hidden_dim;

        if (accumulated >= threshold) {
            float remaining = threshold - (accumulated - alpha);
            for (int d = 0; d < hidden_dim; ++d) {
                current_embed[d] += remaining * enc_frame[d];
            }

            // Fire: emit this embedding
            result.acoustic_embeds.insert(result.acoustic_embeds.end(),
                                           current_embed.begin(), current_embed.end());
            result.n_tokens++;

            // Start new with leftover
            float leftover = accumulated - threshold;
            accumulated = leftover;
            std::fill(current_embed.begin(), current_embed.end(), 0.0f);
            for (int d = 0; d < hidden_dim; ++d) {
                current_embed[d] = leftover * enc_frame[d];
            }
        } else {
            for (int d = 0; d < hidden_dim; ++d) {
                current_embed[d] += alpha * enc_frame[d];
            }
        }
    }

    // Tail handling: if significant accumulated weight remains, fire one more
    if (accumulated > 0.5f && result.n_tokens > 0) {
        result.acoustic_embeds.insert(result.acoustic_embeds.end(),
                                       current_embed.begin(), current_embed.end());
        result.n_tokens++;
    }

    return result;
}

// =====================================================================
// Detokenize: token IDs -> string
// =====================================================================
std::string ParaformerEngine::detokenize(const std::vector<int64_t>& token_ids) {
    std::string result;
    for (int64_t id : token_ids) {
        if (id < 0 || id >= (int64_t)vocab_.size()) continue;
        if (id == BLANK_ID || id == EOS_ID) continue;

        const std::string& tok = vocab_[id];
        // Replace SentencePiece boundary marker (U+2581 = 0xE2 0x96 0x81) with space
        size_t i = 0;
        while (i < tok.size()) {
            if (i + 2 < tok.size() &&
                (unsigned char)tok[i] == 0xE2 &&
                (unsigned char)tok[i+1] == 0x96 &&
                (unsigned char)tok[i+2] == 0x81) {
                result += ' ';
                i += 3;
            } else {
                result += tok[i];
                i++;
            }
        }
    }

    // Trim whitespace
    auto start = result.find_first_not_of(" \t\n\r");
    auto end = result.find_last_not_of(" \t\n\r");
    if (start != std::string::npos) {
        result = result.substr(start, end - start + 1);
    } else {
        result.clear();
    }
    return result;
}

// =====================================================================
// Reset decoder caches to zeros
// =====================================================================
void ParaformerEngine::reset_decoder_caches() {
    decoder_caches_.resize(NUM_DECODER_LAYERS);
    for (int i = 0; i < NUM_DECODER_LAYERS; ++i) {
        decoder_caches_[i].assign(DECODER_HIDDEN_DIM * DECODER_CACHE_TIME, 0.0f);
    }
}

// =====================================================================
// Initialize: load encoder + decoder ONNX models, CMVN, tokenizer
// =====================================================================
bool ParaformerEngine::initialize(const std::string& model_dir) {
    try {
        std::cout << "ParaformerEngine::initialize(" << model_dir << ")" << std::endl;

        std::string encoder_path = model_dir + "/paraformer_encoder.int8.onnx";
        std::string decoder_path = model_dir + "/paraformer_decoder.int8.onnx";
        std::string mvn_path = model_dir + "/am.mvn";
        std::string tokens_path = model_dir + "/paraformer_tokens.json";

        // Check encoder exists
        {
            std::ifstream f(encoder_path);
            if (!f.good()) {
                std::cerr << "Paraformer encoder not found: " << encoder_path << std::endl;
                return false;
            }
        }
        // Check decoder exists
        {
            std::ifstream f(decoder_path);
            if (!f.good()) {
                std::cerr << "Paraformer decoder not found: " << decoder_path << std::endl;
                return false;
            }
        }

        // Session options (same as SenseVoice — mobile-optimized)
        session_options_.SetIntraOpNumThreads(2);
        session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        session_options_.AddConfigEntry("session.use_device_allocator_for_initializers", "1");
        session_options_.DisableMemPattern();

        // Load encoder
        std::cout << "  Loading encoder: " << encoder_path << std::endl;
        encoder_session_ = std::make_unique<Ort::Session>(
            env_, encoder_path.c_str(), session_options_);
        std::cout << "  Encoder loaded" << std::endl;

        // Load decoder (needs separate options to avoid shared state issues)
        Ort::SessionOptions decoder_opts;
        decoder_opts.SetIntraOpNumThreads(2);
        decoder_opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        decoder_opts.AddConfigEntry("session.use_device_allocator_for_initializers", "1");
        decoder_opts.DisableMemPattern();

        std::cout << "  Loading decoder: " << decoder_path << std::endl;
        decoder_session_ = std::make_unique<Ort::Session>(
            env_, decoder_path.c_str(), decoder_opts);
        std::cout << "  Decoder loaded" << std::endl;

        // Load CMVN (560-dim, after LFR)
        if (!load_cmvn(mvn_path)) {
            std::cerr << "Failed to load CMVN from " << mvn_path << std::endl;
            return false;
        }

        // Load tokenizer (JSON array)
        if (!load_tokens_json(tokens_path)) {
            std::cerr << "Failed to load tokens from " << tokens_path << std::endl;
            return false;
        }

        // Initialize decoder caches
        reset_decoder_caches();

        initialized_ = true;
        std::cout << "ParaformerEngine initialized successfully" << std::endl;
        return true;

    } catch (const Ort::Exception& e) {
        std::cerr << "ONNX Runtime error: " << e.what() << std::endl;
        return false;
    } catch (const std::exception& e) {
        std::cerr << "Init error: " << e.what() << std::endl;
        return false;
    }
}

// =====================================================================
// Run encoder ONNX session
//   Input:  speech [1, T_lfr, 560], speech_lengths [1]
//   Output: enc [1, T, 512], enc_len [1], alphas [1, T]
// =====================================================================
ParaformerEngine::EncoderOutput ParaformerEngine::run_encoder(
    const std::vector<float>& lfr_features, int n_lfr_frames)
{
    Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(
        OrtDeviceAllocator, OrtMemTypeCPU);

    // speech [1, T_lfr, 560]
    std::array<int64_t, 3> speech_shape = {1, (int64_t)n_lfr_frames, LFR_FEAT_DIM};
    std::vector<float> speech_data = lfr_features;
    Ort::Value speech_tensor = Ort::Value::CreateTensor<float>(
        mem_info, speech_data.data(), speech_data.size(),
        speech_shape.data(), speech_shape.size());

    // speech_lengths [1]
    std::array<int32_t, 1> length_data = {(int32_t)n_lfr_frames};
    std::array<int64_t, 1> length_shape = {1};
    Ort::Value length_tensor = Ort::Value::CreateTensor<int32_t>(
        mem_info, length_data.data(), 1,
        length_shape.data(), length_shape.size());

    const char* input_names[] = {"speech", "speech_lengths"};
    const char* output_names[] = {"enc", "enc_len", "alphas"};

    std::vector<Ort::Value> inputs;
    inputs.push_back(std::move(speech_tensor));
    inputs.push_back(std::move(length_tensor));

    auto outputs = encoder_session_->Run(
        Ort::RunOptions{nullptr},
        input_names, inputs.data(), 2,
        output_names, 3);

    // Parse outputs
    EncoderOutput enc_out;

    // enc [1, T, 512]
    auto enc_shape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
    enc_out.T = static_cast<int>(enc_shape[1]);
    const float* enc_data = outputs[0].GetTensorData<float>();
    enc_out.enc.assign(enc_data, enc_data + enc_shape[1] * ENCODER_OUTPUT_DIM);

    // enc_len [1]
    enc_out.enc_len = outputs[1].GetTensorData<int32_t>()[0];

    // alphas [1, T]
    const float* alphas_data = outputs[2].GetTensorData<float>();
    auto alphas_shape = outputs[2].GetTensorTypeAndShapeInfo().GetShape();
    int alphas_len = static_cast<int>(alphas_shape[1]);
    enc_out.alphas.assign(alphas_data, alphas_data + alphas_len);

    std::cout << "  Encoder output: T=" << enc_out.T
              << " enc_len=" << enc_out.enc_len
              << " alphas_len=" << alphas_len << std::endl;

    return enc_out;
}

// =====================================================================
// Run decoder ONNX session
//   Input:  enc[1,T,512] + enc_len[1] + acoustic_embeds[1,N,512]
//           + acoustic_embeds_len[1] + 16x in_cache[1,512,10]
//   Output: logits[1,N,8404] + sample_ids[1,N] + 16x out_cache
// =====================================================================
std::vector<int64_t> ParaformerEngine::run_decoder(
    const EncoderOutput& enc_out,
    const CIFResult& cif_result)
{
    if (cif_result.n_tokens <= 0) return {};

    Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(
        OrtDeviceAllocator, OrtMemTypeCPU);

    // Build input tensors
    // enc [1, T, 512]
    std::array<int64_t, 3> enc_shape = {1, (int64_t)enc_out.T, ENCODER_OUTPUT_DIM};
    std::vector<float> enc_data = enc_out.enc;
    Ort::Value enc_tensor = Ort::Value::CreateTensor<float>(
        mem_info, enc_data.data(), enc_data.size(),
        enc_shape.data(), enc_shape.size());

    // enc_len [1]
    std::array<int32_t, 1> enc_len_data = {(int32_t)enc_out.enc_len};
    std::array<int64_t, 1> enc_len_shape = {1};
    Ort::Value enc_len_tensor = Ort::Value::CreateTensor<int32_t>(
        mem_info, enc_len_data.data(), 1,
        enc_len_shape.data(), enc_len_shape.size());

    // acoustic_embeds [1, N, 512]
    int N = cif_result.n_tokens;
    std::array<int64_t, 3> ae_shape = {1, (int64_t)N, ENCODER_OUTPUT_DIM};
    std::vector<float> ae_data = cif_result.acoustic_embeds;
    Ort::Value ae_tensor = Ort::Value::CreateTensor<float>(
        mem_info, ae_data.data(), ae_data.size(),
        ae_shape.data(), ae_shape.size());

    // acoustic_embeds_len [1]
    std::array<int32_t, 1> ae_len_data = {(int32_t)N};
    std::array<int64_t, 1> ae_len_shape = {1};
    Ort::Value ae_len_tensor = Ort::Value::CreateTensor<int32_t>(
        mem_info, ae_len_data.data(), 1,
        ae_len_shape.data(), ae_len_shape.size());

    // 16x in_cache [1, 512, 10]
    std::array<int64_t, 3> cache_shape = {1, DECODER_HIDDEN_DIM, DECODER_CACHE_TIME};
    std::vector<Ort::Value> cache_tensors;
    for (int i = 0; i < NUM_DECODER_LAYERS; ++i) {
        cache_tensors.push_back(Ort::Value::CreateTensor<float>(
            mem_info, decoder_caches_[i].data(), decoder_caches_[i].size(),
            cache_shape.data(), cache_shape.size()));
    }

    // Build input names: enc, enc_len, acoustic_embeds, acoustic_embeds_len, in_cache_0..15
    std::vector<const char*> input_names = {
        "enc", "enc_len", "acoustic_embeds", "acoustic_embeds_len"
    };
    // Cache input names (static storage)
    static const char* cache_input_names[NUM_DECODER_LAYERS] = {
        "in_cache_0", "in_cache_1", "in_cache_2", "in_cache_3",
        "in_cache_4", "in_cache_5", "in_cache_6", "in_cache_7",
        "in_cache_8", "in_cache_9", "in_cache_10", "in_cache_11",
        "in_cache_12", "in_cache_13", "in_cache_14", "in_cache_15"
    };
    for (int i = 0; i < NUM_DECODER_LAYERS; ++i) {
        input_names.push_back(cache_input_names[i]);
    }

    // Build output names: logits, sample_ids, out_cache_0..15
    std::vector<const char*> output_names = {"logits", "sample_ids"};
    static const char* cache_output_names[NUM_DECODER_LAYERS] = {
        "out_cache_0", "out_cache_1", "out_cache_2", "out_cache_3",
        "out_cache_4", "out_cache_5", "out_cache_6", "out_cache_7",
        "out_cache_8", "out_cache_9", "out_cache_10", "out_cache_11",
        "out_cache_12", "out_cache_13", "out_cache_14", "out_cache_15"
    };
    for (int i = 0; i < NUM_DECODER_LAYERS; ++i) {
        output_names.push_back(cache_output_names[i]);
    }

    // Collect all input tensors in order
    std::vector<Ort::Value> all_inputs;
    all_inputs.push_back(std::move(enc_tensor));
    all_inputs.push_back(std::move(enc_len_tensor));
    all_inputs.push_back(std::move(ae_tensor));
    all_inputs.push_back(std::move(ae_len_tensor));
    for (int i = 0; i < NUM_DECODER_LAYERS; ++i) {
        all_inputs.push_back(std::move(cache_tensors[i]));
    }

    // Run decoder
    auto outputs = decoder_session_->Run(
        Ort::RunOptions{nullptr},
        input_names.data(), all_inputs.data(),
        input_names.size(),
        output_names.data(), output_names.size());

    // Read sample_ids [1, N] (already argmax'd by the model)
    const int64_t* sample_ids_data = outputs[1].GetTensorData<int64_t>();
    auto sample_ids_shape = outputs[1].GetTensorTypeAndShapeInfo().GetShape();
    int n_ids = static_cast<int>(sample_ids_shape[1]);
    std::vector<int64_t> token_ids(sample_ids_data, sample_ids_data + n_ids);

    // Update decoder caches from output for streaming
    for (int i = 0; i < NUM_DECODER_LAYERS; ++i) {
        auto& out_cache = outputs[2 + i];
        auto out_shape = out_cache.GetTensorTypeAndShapeInfo().GetShape();
        const float* out_data = out_cache.GetTensorData<float>();
        int out_size = static_cast<int>(out_shape[1] * out_shape[2]);

        // For next call, we only keep the last DECODER_CACHE_TIME columns
        int total_time = static_cast<int>(out_shape[2]);
        int keep_time = std::min(total_time, DECODER_CACHE_TIME);
        int start_col = total_time - keep_time;

        decoder_caches_[i].assign(DECODER_HIDDEN_DIM * DECODER_CACHE_TIME, 0.0f);
        for (int r = 0; r < DECODER_HIDDEN_DIM; ++r) {
            for (int c = 0; c < keep_time; ++c) {
                int dst_col = DECODER_CACHE_TIME - keep_time + c;
                decoder_caches_[i][r * DECODER_CACHE_TIME + dst_col] =
                    out_data[r * total_time + start_col + c];
            }
        }
    }

    std::cout << "  Decoder output: " << n_ids << " tokens" << std::endl;
    return token_ids;
}

// =====================================================================
// Transcribe: offline (no metrics)
// =====================================================================
std::string ParaformerEngine::transcribe(const std::vector<float>& pcm_data) {
    InferenceMetrics unused;
    return transcribe(pcm_data, unused);
}

// =====================================================================
// Transcribe: offline with metrics
//   PCM -> Fbank(80) -> LFR(7,6) -> CMVN(560) -> Encoder -> CIF -> Decoder
// =====================================================================
std::string ParaformerEngine::transcribe(const std::vector<float>& pcm_data,
                                          InferenceMetrics& metrics) {
    if (!initialized_) return "Error: Engine not initialized";

    uint64_t t_total_start = mach_absolute_time();
    metrics.memory_bytes_before = get_rss_bytes();
    metrics.audio_duration_ms = static_cast<double>(pcm_data.size()) / 16000.0 * 1000.0;

    std::cout << "Paraformer transcribe: " << pcm_data.size() << " samples ("
              << metrics.audio_duration_ms << "ms)" << std::endl;

    try {
        // 1. Fbank extraction: PCM -> [n_frames, 80]
        uint64_t t0 = mach_absolute_time();
        int n_fbank_frames = 0;
        std::vector<float> fbank = feature_extractor_.process(pcm_data, n_fbank_frames);
        uint64_t t1 = mach_absolute_time();
        metrics.fbank_time_ms = mach_time_to_ms(t0, t1);
        std::cout << "  Fbank: " << n_fbank_frames << " x 80" << std::endl;

        // 2. LFR: [n_frames, 80] -> [n_lfr_frames, 560]
        uint64_t t2 = mach_absolute_time();
        int n_lfr_frames = 0;
        std::vector<float> lfr_features = apply_lfr(fbank, n_fbank_frames, 80, n_lfr_frames);
        std::cout << "  LFR: " << n_lfr_frames << " x 560" << std::endl;

        // 3. CMVN on 560-dim LFR features
        apply_cmvn(lfr_features, n_lfr_frames, LFR_FEAT_DIM);
        uint64_t t3 = mach_absolute_time();
        metrics.lfr_cmvn_time_ms = mach_time_to_ms(t2, t3);

        // 4. Encoder forward
        uint64_t t_enc_start = mach_absolute_time();
        EncoderOutput enc_out = run_encoder(lfr_features, n_lfr_frames);
        uint64_t t_enc_end = mach_absolute_time();

        // 5. CIF predictor
        CIFResult cif_result = run_cif(
            enc_out.enc.data(), enc_out.alphas.data(),
            enc_out.T, ENCODER_OUTPUT_DIM);
        std::cout << "  CIF: " << cif_result.n_tokens << " tokens predicted" << std::endl;

        if (cif_result.n_tokens <= 0) {
            metrics.onnx_forward_time_ms = mach_time_to_ms(t_enc_start, t_enc_end);
            metrics.decode_time_ms = 0;
            uint64_t t_total_end = mach_absolute_time();
            metrics.total_inference_ms = mach_time_to_ms(t_total_start, t_total_end);
            metrics.memory_bytes_after = get_rss_bytes();
            metrics.rtf = 0;
            return "";
        }

        // 6. Decoder forward
        reset_decoder_caches();
        uint64_t t_dec_start = mach_absolute_time();
        std::vector<int64_t> token_ids = run_decoder(enc_out, cif_result);
        uint64_t t_dec_end = mach_absolute_time();

        metrics.onnx_forward_time_ms = mach_time_to_ms(t_enc_start, t_dec_end);
        metrics.decode_time_ms = mach_time_to_ms(t_dec_start, t_dec_end);

        // 7. Detokenize
        std::string result = detokenize(token_ids);
        std::cout << "  Result: " << result << std::endl;

        uint64_t t_total_end = mach_absolute_time();
        metrics.memory_bytes_after = get_rss_bytes();
        metrics.total_inference_ms = mach_time_to_ms(t_total_start, t_total_end);
        metrics.rtf = (metrics.audio_duration_ms > 0)
            ? metrics.total_inference_ms / metrics.audio_duration_ms : 0.0;

        return result;

    } catch (const Ort::Exception& e) {
        std::cerr << "ONNX Runtime error: " << e.what() << std::endl;
        return std::string("ONNX Error: ") + e.what();
    } catch (const std::exception& e) {
        std::cerr << "Paraformer error: " << e.what() << std::endl;
        return std::string("Error: ") + e.what();
    }
}

// =====================================================================
// Streaming interface
// =====================================================================
void ParaformerEngine::start_stream() {
    stream_active_ = true;
    stream_pcm_buffer_.clear();
    reset_decoder_caches();
}

std::string ParaformerEngine::feed_chunk(const std::vector<float>& pcm_chunk) {
    if (!stream_active_ || !initialized_) return "";

    stream_pcm_buffer_.insert(stream_pcm_buffer_.end(),
                               pcm_chunk.begin(), pcm_chunk.end());

    // Need at least 500ms of audio for meaningful results
    if (stream_pcm_buffer_.size() < 8000) return "";

    try {
        // Run full pipeline on accumulated audio (pseudo-streaming for now)
        int n_frames = 0;
        std::vector<float> fbank = feature_extractor_.process(stream_pcm_buffer_, n_frames);
        if (n_frames <= 0) return "";

        int n_lfr_frames = 0;
        std::vector<float> lfr_features = apply_lfr(fbank, n_frames, 80, n_lfr_frames);
        apply_cmvn(lfr_features, n_lfr_frames, LFR_FEAT_DIM);

        EncoderOutput enc_out = run_encoder(lfr_features, n_lfr_frames);
        CIFResult cif_result = run_cif(
            enc_out.enc.data(), enc_out.alphas.data(),
            enc_out.T, ENCODER_OUTPUT_DIM);

        if (cif_result.n_tokens <= 0) return "";

        // Note: don't update caches for intermediate results
        std::vector<std::vector<float>> saved_caches = decoder_caches_;
        std::vector<int64_t> token_ids = run_decoder(enc_out, cif_result);
        decoder_caches_ = saved_caches;  // restore caches

        return detokenize(token_ids);

    } catch (...) {
        return "";
    }
}

std::string ParaformerEngine::end_stream(InferenceMetrics& metrics) {
    if (!stream_active_) return "";
    stream_active_ = false;

    if (stream_pcm_buffer_.empty()) return "";

    std::string result = transcribe(stream_pcm_buffer_, metrics);
    stream_pcm_buffer_.clear();
    reset_decoder_caches();
    return result;
}
