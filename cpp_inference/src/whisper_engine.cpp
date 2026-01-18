#include "whisper_engine.h"
#include <iostream>
#include <numeric>
#include <limits>
#include <vector>
#include <chrono>
#include <cstring>
#include "coreml_provider_factory.h"

// --- 🛠️ FP16 Helper Functions (Pure C++ / IEEE 754) ---
// 这种实现方式不依赖 vDSP 或特殊的 _Float16 类型，兼容性最强 (Intel/M1 均可)
// This implementation relies on IEEE 754 bit manipulation, works on Intel & M1 without vDSP dependency for conversion.

// Helper: Float32 to Float16 bits
uint16_t float_to_half_bits(float f) {
    uint32_t x;
    std::memcpy(&x, &f, 4);
    uint32_t sign = (x >> 31) & 0x1;
    uint32_t exp = (x >> 23) & 0xFF;
    uint32_t mant = x & 0x7FFFFF;
    uint16_t h = 0;

    if (exp == 0) {
        h = (uint16_t)(sign << 15);
    } else if (exp == 255) {
        h = (uint16_t)((sign << 15) | 0x7C00 | (mant ? 1 : 0));
    } else {
        int new_exp = exp - 127 + 15;
        if (new_exp >= 31) {
            h = (uint16_t)((sign << 15) | 0x7C00);
        } else if (new_exp <= 0) {
            h = (uint16_t)(sign << 15);
        } else {
            h = (uint16_t)((sign << 15) | (new_exp << 10) | (mant >> 13));
        }
    }
    return h;
}

// Helper: Float16 bits to Float32
float half_bits_to_float(uint16_t h) {
    uint32_t sign = (h >> 15) & 0x1;
    uint32_t exp = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    uint32_t f = 0;

    if (exp == 0) {
        if (mant == 0) {
            f = (sign << 31);
        } else {
            f = (sign << 31) | ((mant + 0) << 13); // Subnormal approximation
        }
    } else if (exp == 31) {
        f = (sign << 31) | 0xFF000000 | (mant ? 0x400000 : 0);
    } else {
        f = (sign << 31) | ((exp - 15 + 127) << 23) | (mant << 13);
    }
    
    float result;
    std::memcpy(&result, &f, 4);
    return result;
}

// 1. Batch Convert Float32 -> Ort::Float16_t
std::vector<Ort::Float16_t> float2half(const std::vector<float>& input) {
    std::vector<Ort::Float16_t> output(input.size());
    for (size_t i = 0; i < input.size(); ++i) {
        uint16_t h_bits = float_to_half_bits(input[i]);
        std::memcpy(&output[i], &h_bits, sizeof(uint16_t));
    }
    return output;
}

// 2. Batch Convert Ort::Float16_t -> Float32
// We implement this loop manually to replace vDSP_vhtof
void half2float_array(const Ort::Float16_t* input, float* output, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        uint16_t h_bits;
        std::memcpy(&h_bits, &input[i], sizeof(uint16_t));
        output[i] = half_bits_to_float(h_bits);
    }
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
                               const std::string& vocab_path) {
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

    // 1. Feature Extraction (Float32)
    std::cout << "🧮 Extracting features..." << std::endl;
    std::vector<float> enc_input_data_fp32 = feature_extractor_.process(pcm_data);

    // 🔥 Convert to FP16 for Model Input (Using Helper)
    std::vector<Ort::Float16_t> enc_input_data_fp16 = float2half(enc_input_data_fp32);

    // ==========================================
    // 2. Run Encoder (FP16 Input)
    // ==========================================
    std::cout << "🚀 Running Encoder..." << std::endl;
    
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    std::vector<int64_t> enc_input_shape = {INPUT_BATCH, N_MELS, N_FRAMES};
    size_t enc_input_size = INPUT_BATCH * N_MELS * N_FRAMES;

    // Create Tensor with Float16_t
    Ort::Value enc_input_tensor = Ort::Value::CreateTensor<Ort::Float16_t>(
        memory_info, enc_input_data_fp16.data(), enc_input_size, enc_input_shape.data(), enc_input_shape.size()
    );

    const char* enc_input_names[] = {"mel"};
    const char* enc_output_names[] = {"output"};

    auto audio_features = encoder_session_->Run(
        Ort::RunOptions{nullptr},
        enc_input_names, &enc_input_tensor, 1,
        enc_output_names, 1
    );

    // ==========================================
    // 3. Initialize Decoder Loop
    // ==========================================
    std::cout << "🗣️  Start Decoding..." << std::endl;
    std::string full_transcription = "";
    
    std::vector<int64_t> generated_tokens;
    generated_tokens.push_back(SOT_TOKEN);

    // Prepare Input Vector
    std::vector<Ort::Value> input_tensors;
    input_tensors.push_back(Ort::Value(nullptr));
    input_tensors.push_back(std::move(audio_features[0]));

    const char* dec_input_names[] = {"tokens", "audio_features"};
    const char* dec_output_names[] = {"logits"};

    int max_steps = 100;
    
    // Pre-allocate buffer for logits
    int64_t vocab_size_guess = 51865;
    std::vector<float> logits_fp32_buffer(vocab_size_guess);

    for (int step = 0; step < max_steps; step++) {
        size_t current_seq_len = generated_tokens.size();
        
        std::vector<int64_t> token_input_shape = {1, static_cast<int64_t>(current_seq_len)};
        
        Ort::Value token_tensor = Ort::Value::CreateTensor<int64_t>(
            memory_info,
            generated_tokens.data(),
            current_seq_len,
            token_input_shape.data(),
            token_input_shape.size()
        );

        input_tensors[0] = std::move(token_tensor);

        // Run Decoder
        auto dec_results = decoder_session_->Run(
            Ort::RunOptions{nullptr},
            dec_input_names,
            input_tensors.data(),
            2,
            dec_output_names,
            1
        );

        // 🔥 Get Logits as FP16 (Raw Pointer)
        Ort::Float16_t* output_data_fp16 = dec_results[0].GetTensorMutableData<Ort::Float16_t>();
        auto shape_info = dec_results[0].GetTensorTypeAndShapeInfo().GetShape();
        int64_t vocab_size = shape_info[2];
        int64_t seq_len = shape_info[1];

        // Pointer to the last timestamp
        Ort::Float16_t* last_logits_ptr = output_data_fp16 + (seq_len - 1) * vocab_size;
        
        if (logits_fp32_buffer.size() < vocab_size) logits_fp32_buffer.resize(vocab_size);

        // 🔥 Batch Convert FP16 -> Float32 using Helper (No vDSP)
        half2float_array(last_logits_ptr, logits_fp32_buffer.data(), vocab_size);

        // Greedy Search on FP32 Data
        int next_token_id = 0;
        float max_val = -std::numeric_limits<float>::infinity();

        for (int i = 0; i < vocab_size; i++) {
            if (logits_fp32_buffer[i] > max_val) {
                max_val = logits_fp32_buffer[i];
                next_token_id = i;
            }
        }

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
