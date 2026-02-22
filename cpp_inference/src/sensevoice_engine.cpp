#include "sensevoice_engine.h"
#include <iostream>
#include <sstream>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

// =====================================================================
// SenseVoiceEngine — Non-autoregressive encoder-only ASR
//
// Single forward pass:
//   Input:  x [1, T_lfr, 560]   float32   (LFR + CMVN features)
//           x_length [1]         int32     (number of LFR frames)
//           language [1]         int32     (language ID, e.g. 3 = zh)
//           text_norm [1]        int32     (ITN flag, e.g. 14 = with_itn)
//   Output: logits [1, T_out, 25055] float32 (CTC output)
// =====================================================================

SenseVoiceEngine::SenseVoiceEngine()
    : env_(ORT_LOGGING_LEVEL_WARNING, "SenseVoice")
{
}

SenseVoiceEngine::~SenseVoiceEngine() = default;

// -------- Parse CSV string from model metadata --------
std::vector<float> SenseVoiceEngine::parse_csv(const std::string& csv) {
    std::vector<float> values;
    std::stringstream ss(csv);
    std::string item;
    while (std::getline(ss, item, ',')) {
        try {
            values.push_back(std::stof(item));
        } catch (...) {
            // skip malformed entries
        }
    }
    return values;
}

// -------- Load CMVN + params from ONNX model metadata --------
bool SenseVoiceEngine::load_metadata() {
    if (!session_) return false;

    Ort::AllocatorWithDefaultOptions allocator;
    auto metadata = session_->GetModelMetadata();

    // Helper lambda to read a metadata key
    auto read_key = [&](const char* key) -> std::string {
        try {
            auto value = metadata.LookupCustomMetadataMapAllocated(key, allocator);
            const char* raw = value.get();
            if (!raw) return "";
            return std::string(raw);
        } catch (...) {
            return "";
        }
    };

    // CMVN parameters (critical)
    std::string neg_mean_str = read_key("neg_mean");
    std::string inv_stddev_str = read_key("inv_stddev");

    if (neg_mean_str.empty() || inv_stddev_str.empty()) {
        std::cerr << "❌ Failed to read CMVN from model metadata" << std::endl;
        return false;
    }

    neg_mean_ = parse_csv(neg_mean_str);
    inv_stddev_ = parse_csv(inv_stddev_str);

    std::cout << "  CMVN loaded: neg_mean size=" << neg_mean_.size()
              << " inv_stddev size=" << inv_stddev_.size() << std::endl;

    if (neg_mean_.size() != 560 || inv_stddev_.size() != 560) {
        std::cerr << "❌ CMVN dimension mismatch: expected 560, got "
                  << neg_mean_.size() << "/" << inv_stddev_.size() << std::endl;
        return false;
    }

    // Optional metadata keys
    auto try_int = [&](const char* key, int& target) {
        std::string val = read_key(key);
        if (!val.empty()) {
            try { target = std::stoi(val); } catch (...) {}
        }
    };

    try_int("lfr_window_size", lfr_window_size_);
    try_int("lfr_window_shift", lfr_window_shift_);
    try_int("blank_id", blank_id_);
    try_int("vocab_size", vocab_size_);

    std::cout << "  Model params: lfr_window=" << lfr_window_size_
              << " lfr_shift=" << lfr_window_shift_
              << " blank_id=" << blank_id_
              << " vocab_size=" << vocab_size_ << std::endl;

    return true;
}

// =====================================================================
// Initialize: load model + tokens
// =====================================================================
bool SenseVoiceEngine::initialize(const std::string& model_path,
                                   const std::string& tokens_path)
{
    try {
        std::cout << "🔧 SenseVoiceEngine::initialize()" << std::endl;

        // Session options: CPU only, no CoreML
        session_options_.SetIntraOpNumThreads(2);
        session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        // Use device allocator (not arena) to avoid memory accumulation
        session_options_.AddConfigEntry("session.use_device_allocator_for_initializers", "1");

        // Disable memory pattern optimization to reduce peak memory
        session_options_.DisableMemPattern();

        std::cout << "  Loading ONNX model: " << model_path << std::endl;
        session_ = std::make_unique<Ort::Session>(env_, model_path.c_str(), session_options_);
        std::cout << "  ✅ Model loaded" << std::endl;

        // Load metadata (CMVN, LFR params, etc.)
        if (!load_metadata()) {
            std::cerr << "❌ Failed to load model metadata" << std::endl;
            return false;
        }

        // Load tokenizer
        std::cout << "  Loading tokens: " << tokens_path << std::endl;
        if (!tokenizer_.load(tokens_path)) {
            std::cerr << "❌ Failed to load tokens" << std::endl;
            return false;
        }

        std::cout << "✅ SenseVoiceEngine initialized successfully" << std::endl;
        return true;

    } catch (const Ort::Exception& e) {
        std::cerr << "❌ ONNX Runtime error: " << e.what() << std::endl;
        return false;
    } catch (const std::exception& e) {
        std::cerr << "❌ Init error: " << e.what() << std::endl;
        return false;
    }
}

// =====================================================================
// LFR (Low Frame Rate): stack lfr_window_size frames, subsample by lfr_window_shift
//   Input:  fbank [n_frames, 80]
//   Output: lfr_features [out_frames, 560]
// =====================================================================
std::vector<float> SenseVoiceEngine::apply_lfr(const std::vector<float>& fbank,
                                                int n_frames, int n_mels,
                                                int& out_frames)
{
    const int W = lfr_window_size_;  // 7
    const int S = lfr_window_shift_; // 6
    const int feat_dim = W * n_mels; // 7 * 80 = 560

    // Number of output frames
    out_frames = (n_frames - W) / S + 1;
    if (out_frames <= 0) {
        // If audio too short, pad input frames and produce 1 output frame
        out_frames = 1;
        std::vector<float> result(feat_dim, 0.0f);
        for (int j = 0; j < std::min(n_frames, W); ++j) {
            std::copy(fbank.begin() + j * n_mels,
                      fbank.begin() + j * n_mels + n_mels,
                      result.begin() + j * n_mels);
        }
        return result;
    }

    std::vector<float> result(out_frames * feat_dim);

    for (int i = 0; i < out_frames; ++i) {
        int start_frame = i * S;
        // Concatenate W consecutive frames (each 80-dim) → 560-dim
        for (int j = 0; j < W; ++j) {
            int src_frame = start_frame + j;
            if (src_frame < n_frames) {
                std::copy(fbank.begin() + src_frame * n_mels,
                          fbank.begin() + src_frame * n_mels + n_mels,
                          result.begin() + i * feat_dim + j * n_mels);
            }
            // else: zero-padded (already 0 from initialization)
        }
    }

    return result;
}

// =====================================================================
// CMVN: feature = (feature + neg_mean) * inv_stddev
//   Applied in-place on [n_frames, 560] features
// =====================================================================
void SenseVoiceEngine::apply_cmvn(std::vector<float>& features,
                                   int n_frames, int feat_dim)
{
    if ((int)neg_mean_.size() != feat_dim || (int)inv_stddev_.size() != feat_dim) {
        throw std::runtime_error("CMVN dimension mismatch");
    }

    for (int i = 0; i < n_frames; ++i) {
        float* frame = features.data() + i * feat_dim;
        for (int j = 0; j < feat_dim; ++j) {
            frame[j] = (frame[j] + neg_mean_[j]) * inv_stddev_[j];
        }
    }
}

// =====================================================================
// CTC greedy decode: argmax per timestep, collapse repeats, skip blank
// =====================================================================
std::string SenseVoiceEngine::ctc_greedy_decode(const float* logits,
                                                 int T, int vocab_size)
{
    std::vector<int> token_ids;
    int prev_token = blank_id_;

    for (int t = 0; t < T; ++t) {
        const float* row = logits + t * vocab_size;

        // Argmax
        int best_id = 0;
        float best_val = row[0];
        for (int v = 1; v < vocab_size; ++v) {
            if (row[v] > best_val) {
                best_val = row[v];
                best_id = v;
            }
        }

        // Collapse: skip blank and repeated tokens
        if (best_id != blank_id_ && best_id != prev_token) {
            token_ids.push_back(best_id);
        }
        prev_token = best_id;
    }

    // Decode token IDs to string
    std::string result = tokenizer_.decode(token_ids);

    // Trim leading/trailing whitespace
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
// Transcribe: full pipeline
//   PCM → fbank → LFR → CMVN → ONNX forward → CTC decode
// =====================================================================
std::string SenseVoiceEngine::transcribe(const std::vector<float>& pcm_data)
{
    if (!session_) {
        return "Error: Model not loaded";
    }

    try {
        std::cout << "🧠 SenseVoice transcribe: " << pcm_data.size() << " samples" << std::endl;

        // 1. Feature extraction: PCM → [n_frames, 80] log-fbank
        int n_fbank_frames = 0;
        std::vector<float> fbank = feature_extractor_.process(pcm_data, n_fbank_frames);
        std::cout << "  Fbank: " << n_fbank_frames << " x 80" << std::endl;

        // 2. LFR: [n_frames, 80] → [n_lfr_frames, 560]
        int n_lfr_frames = 0;
        std::vector<float> lfr_features = apply_lfr(fbank, n_fbank_frames, 80, n_lfr_frames);
        std::cout << "  LFR: " << n_lfr_frames << " x 560" << std::endl;

        // 3. CMVN normalization (in-place)
        apply_cmvn(lfr_features, n_lfr_frames, 560);

        // 4. Prepare ONNX inputs
        Ort::AllocatorWithDefaultOptions allocator;
        Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(
            OrtDeviceAllocator, OrtMemTypeCPU);

        // Input 0: x [1, T_lfr, 560]
        std::array<int64_t, 3> x_shape = {1, (int64_t)n_lfr_frames, 560};
        Ort::Value x_tensor = Ort::Value::CreateTensor<float>(
            mem_info, lfr_features.data(), lfr_features.size(),
            x_shape.data(), x_shape.size());

        // Input 1: x_length [1]
        std::array<int32_t, 1> x_length_data = {(int32_t)n_lfr_frames};
        std::array<int64_t, 1> x_length_shape = {1};
        Ort::Value x_length_tensor = Ort::Value::CreateTensor<int32_t>(
            mem_info, x_length_data.data(), 1,
            x_length_shape.data(), x_length_shape.size());

        // Input 2: language [1]  (zh = 3)
        std::array<int32_t, 1> language_data = {(int32_t)lang_zh_id_};
        std::array<int64_t, 1> language_shape = {1};
        Ort::Value language_tensor = Ort::Value::CreateTensor<int32_t>(
            mem_info, language_data.data(), 1,
            language_shape.data(), language_shape.size());

        // Input 3: text_norm [1]  (with_itn = 14)
        std::array<int32_t, 1> text_norm_data = {(int32_t)with_itn_id_};
        std::array<int64_t, 1> text_norm_shape = {1};
        Ort::Value text_norm_tensor = Ort::Value::CreateTensor<int32_t>(
            mem_info, text_norm_data.data(), 1,
            text_norm_shape.data(), text_norm_shape.size());

        // Set up input/output names
        const char* input_names[] = {"x", "x_length", "language", "text_norm"};
        const char* output_names[] = {"logits"};

        // Collect input tensors
        std::vector<Ort::Value> input_tensors;
        input_tensors.push_back(std::move(x_tensor));
        input_tensors.push_back(std::move(x_length_tensor));
        input_tensors.push_back(std::move(language_tensor));
        input_tensors.push_back(std::move(text_norm_tensor));

        // 5. Run inference (single forward pass!)
        std::cout << "  Running ONNX inference..." << std::endl;
        auto outputs = session_->Run(
            Ort::RunOptions{nullptr},
            input_names, input_tensors.data(), 4,
            output_names, 1);

        // 6. Read output logits [1, T_out, vocab_size]
        auto& logits_tensor = outputs[0];
        auto logits_info = logits_tensor.GetTensorTypeAndShapeInfo();
        auto logits_shape = logits_info.GetShape();

        int T_out = static_cast<int>(logits_shape[1]);
        int V = static_cast<int>(logits_shape[2]);
        std::cout << "  Logits shape: [1, " << T_out << ", " << V << "]" << std::endl;

        const float* logits_data = logits_tensor.GetTensorData<float>();

        // 7. CTC greedy decode
        std::string result = ctc_greedy_decode(logits_data, T_out, V);
        std::cout << "  Result: " << result << std::endl;

        return result;

    } catch (const Ort::Exception& e) {
        std::cerr << "❌ ONNX Runtime error: " << e.what() << std::endl;
        return std::string("ONNX Error: ") + e.what();
    } catch (const std::exception& e) {
        std::cerr << "❌ Transcribe error: " << e.what() << std::endl;
        return std::string("Error: ") + e.what();
    }
}
