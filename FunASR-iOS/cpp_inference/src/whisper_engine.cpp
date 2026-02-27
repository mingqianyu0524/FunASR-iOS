#include "whisper_engine.h"
#include <iostream>
#include <numeric>
#include <limits>
#include <vector>
#include <chrono>
#include <cstring>
#include "coreml_provider_factory.h"

// ------------------------------------------------
// Helper: convert a float32 vector to Ort::Float16_t vector
static std::vector<Ort::Float16_t> float_to_fp16(const std::vector<float>& src) {
    std::vector<Ort::Float16_t> dst(src.size());
    for (size_t i = 0; i < src.size(); ++i) {
        dst[i] = Ort::Float16_t(src[i]);
    }
    return dst;
}

// ------------------------------------------------

WhisperEngine::WhisperEngine()
    : env_(ORT_LOGGING_LEVEL_WARNING, "WhisperEngine")
{
    session_options_.SetIntraOpNumThreads(4);
    session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_BASIC);
}

WhisperEngine::~WhisperEngine() {
}

bool WhisperEngine::initialize(const std::string& encoder_path,
                               const std::string& decoder_path,
                               const std::string& vocab_path,
                               const std::string& embedder_path) {
    try {
        std::cout << "⏳ WhisperEngine: Loading models..." << std::endl;

        // --- CoreML Config ---
        uint32_t coreml_flags = 0;
        coreml_flags |= COREML_FLAG_ONLY_ENABLE_DEVICE_WITH_ANE;

        const OrtApi g_ort = Ort::GetApi();
        OrtStatus* status = OrtSessionOptionsAppendExecutionProvider_CoreML(session_options_, coreml_flags);

        if (status != nullptr) {
            const char* msg = g_ort.GetErrorMessage(status);
            std::cerr << "⚠️ Warning: Failed to enable CoreML: " << msg << std::endl;
            g_ort.ReleaseStatus(status);
        } else {
            std::cout << "✅ CoreML Execution Provider enabled (ANE Only)!" << std::endl;
        }

        auto start_enc = std::chrono::high_resolution_clock::now();
        encoder_session_ = std::make_unique<Ort::Session>(env_, encoder_path.c_str(), session_options_);
        auto end_enc = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed_enc = end_enc - start_enc;
        std::cout << "✅ Encoder Loaded. Time: " << elapsed_enc.count() << "s" << std::endl;

        auto start_dec = std::chrono::high_resolution_clock::now();
        decoder_session_ = std::make_unique<Ort::Session>(env_, decoder_path.c_str(), session_options_);
        auto end_dec = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed_dec = end_dec - start_dec;
        std::cout << "✅ Decoder Loaded. Time: " << elapsed_dec.count() << "s" << std::endl;

        if (!tokenizer_.load(vocab_path)) {
            std::cerr << "❌ WhisperEngine: Failed to load vocab: " << vocab_path << std::endl;
            return false;
        }

        // Note: embedder_ not needed — embedding layer is baked into the decoder model.
        // We pass raw token IDs (int64) to the decoder instead.

        std::cout << "✅ WhisperEngine: Initialized successfully." << std::endl;
        return true;

    } catch (const Ort::Exception& e) {
        std::cerr << "❌ ONNX Runtime Error: " << e.what() << std::endl;
        return false;
    }
}

std::string WhisperEngine::transcribe(const std::vector<float>& pcm_data) {
    if (!encoder_session_ || !decoder_session_) {
        std::cerr << "❌ Engine not initialized!" << std::endl;
        return "";
    }

    if (pcm_data.empty()) return "";

    // Use OrtDeviceAllocator instead of OrtArenaAllocator to avoid
    // memory accumulation across multiple inference runs.
    // The arena allocator grows but never releases memory, causing OOM on repeated calls.
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);

    // ==========================================
    // 1. Feature Extraction & Encoder
    // ==========================================
    std::cout << "🧮 Extracting features..." << std::endl;

    // Scope the mel buffers so they are freed before decoder runs
    std::vector<Ort::Value> audio_features;
    {
        std::vector<float> mel_data_f32 = feature_extractor_.process(pcm_data);

        // Convert mel features from float32 to float16 (model expects fp16 input)
        std::vector<Ort::Float16_t> mel_data_f16 = float_to_fp16(mel_data_f32);

        // Free f32 immediately — no longer needed
        mel_data_f32.clear();
        mel_data_f32.shrink_to_fit();

        std::cout << "🚀 Running Encoder..." << std::endl;

        // Shape: [1, 80, 3000]
        std::vector<int64_t> enc_input_shape = {1, N_MELS, N_FRAMES};

        Ort::Value enc_input_tensor = Ort::Value::CreateTensor<Ort::Float16_t>(
            memory_info,
            mel_data_f16.data(),
            mel_data_f16.size(),
            enc_input_shape.data(),
            enc_input_shape.size()
        );

        const char* enc_input_names[] = {"mel"};
        const char* enc_output_names[] = {"output"};

        audio_features = encoder_session_->Run(
            Ort::RunOptions{nullptr},
            enc_input_names, &enc_input_tensor, 1,
            enc_output_names, 1
        );
        // mel_data_f16 freed here when scope exits
    }

    std::cout << "✅ Encoder done." << std::endl;

    // ==========================================
    // 2. Decoder Loop
    // ==========================================
    std::cout << "🗣️ Start Decoding..." << std::endl;
    std::string full_transcription = "";

    std::vector<int64_t> generated_tokens;
    generated_tokens.push_back(SOT_TOKEN);

    // Decoder inputs: tokens (int64), audio_features (fp16, from encoder output)
    std::vector<Ort::Value> input_tensors;
    input_tensors.reserve(2);
    input_tensors.push_back(Ort::Value(nullptr)); // placeholder for tokens tensor
    input_tensors.push_back(std::move(audio_features[0])); // encoder output (fp16)

    const char* dec_input_names[] = {"tokens", "audio_features"};
    const char* dec_output_names[] = {"logits"};

    int max_steps = 100;

    for (int step = 0; step < max_steps; step++) {
        // A. Create token tensor: shape [1, seq_len], dtype int64
        std::vector<int64_t> token_shape = {INPUT_BATCH, static_cast<int64_t>(generated_tokens.size())};

        Ort::Value token_tensor = Ort::Value::CreateTensor<int64_t>(
            memory_info,
            generated_tokens.data(),
            generated_tokens.size(),
            token_shape.data(),
            token_shape.size()
        );

        input_tensors[0] = std::move(token_tensor);

        // B. Run Decoder
        auto dec_results = decoder_session_->Run(
            Ort::RunOptions{nullptr},
            dec_input_names,
            input_tensors.data(),
            2,
            dec_output_names,
            1
        );

        // C. Process output logits (fp16)
        auto shape_info = dec_results[0].GetTensorTypeAndShapeInfo().GetShape();
        int64_t seq_len = shape_info[1];
        int64_t vocab_size = shape_info[2];

        Ort::Float16_t* logits_fp16 = dec_results[0].GetTensorMutableData<Ort::Float16_t>();

        // Get logits for the last token position only
        Ort::Float16_t* last_logits_ptr = logits_fp16 + (seq_len - 1) * vocab_size;

        // Greedy Search (convert fp16 -> float on the fly for comparison)
        int next_token_id = 0;
        float max_val = -std::numeric_limits<float>::infinity();

        for (int i = 0; i < vocab_size; i++) {
            float val = last_logits_ptr[i].ToFloat();
            if (val > max_val) {
                max_val = val;
                next_token_id = i;
            }
        }

        // dec_results goes out of scope here, freeing logits memory

        if (next_token_id == EOT_TOKEN) {
            std::cout << "\n🛑 Reached <EOT>" << std::endl;
            break;
        }

        generated_tokens.push_back(next_token_id);

        std::string token_text = tokenizer_.decode(next_token_id);
        std::cout << token_text << std::flush;
        full_transcription += token_text;
    }

    return full_transcription;
}
