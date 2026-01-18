#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <memory>
#include <limits>
#include <chrono>
#include "onnxruntime/onnxruntime_cxx_api.h" // 确保路径正确
#include "whisper_tokenizer.h"
#include "audio_loader.h"
#include "feature_extractor.h"

// 设定输入大小
const int INPUT_SIZE = 1 * 80 * 3000;
const std::vector<int64_t> enc_input_shape{1, 80, 3000};
FeatureExtractor feature_extractor;

int test_main(int argc, char* argv[])
{
    // ==========================================
    // 1. 初始化环境与加载模型
    // ==========================================
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "WhisperInference");
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_BASIC); // 降低优化级别

    std::cout << "⏳ Loading Encoder..." << std::endl;
    auto start_enc = std::chrono::high_resolution_clock::now();
    
    std::unique_ptr<Ort::Session> encoder_session = std::make_unique<Ort::Session>(env, "../models/encoder_int8.onnx", session_options);
    
    auto end_enc = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_enc = end_enc - start_enc;
    std::cout << "✅ Encoder Loaded. Time: " << elapsed_enc.count() << "s" << std::endl;

    std::cout << "⏳ Loading Decoder..." << std::endl;
    auto start_dec = std::chrono::high_resolution_clock::now();
    
    std::unique_ptr<Ort::Session> decoder_session = std::make_unique<Ort::Session>(env, "../models/decoder_int8.onnx", session_options);
    
    auto end_dec = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_dec = end_dec - start_dec;
    std::cout << "✅ Decoder Loaded. Time: " << elapsed_dec.count() << "s" << std::endl;

    // ==========================================
    // 2. 加载 Tokenizer
    // ==========================================
    WhisperTokenizer tokenizer;
    if (!tokenizer.load("../models/vocab.json")) {
        std::cerr << "❌ Failed to load vocab.json" << std::endl;
        return 1;
    }

    // ==========================================
    // 3. 加载输入数据 (Mel Spectrogram)
    // ==========================================
    if (argc < 2) {
        std::cerr << "Usage: ./whisper_inference <path_to_mel_bin>" << std::endl;
        return 1;
    }
    std::string data_path = argv[1];

    // 1. Load WAV -> PCM (30s, 16k, Mono)
    // Assuming AudioLoader::load is static; if not, instantiate AudioLoader first
    std::cout << "🎤 Loading audio file: " << data_path << std::endl;
    std::vector<float> pcm_data = AudioLoader::load(data_path);

    // Simple check
    if (pcm_data.empty()) {
        std::cerr << "❌ Failed to load audio." << std::endl;
        return 1;
    }

    // 2. Feature Extraction -> Mel Spectrogram (80 x 3000)
    std::cout << "🧮 Extracting features..." << std::endl;
    std::vector<float> enc_input_data = feature_extractor.process(pcm_data);

    // ==========================================
    // 4. 运行 Encoder
    // ==========================================
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    Ort::Value enc_input_tensor = Ort::Value::CreateTensor<float>(
        memory_info, enc_input_data.data(), INPUT_SIZE, enc_input_shape.data(), enc_input_shape.size()
    );

    const char* enc_input_names[] = {"mel"};
    const char* enc_output_names[] = {"output"}; // 根据你的 export 脚本，这里可能是 "last_hidden_state" 或 "output"，请确保一致

    std::cout << "🚀 Running Encoder..." << std::endl;
    auto audio_features = encoder_session->Run(
        Ort::RunOptions{nullptr}, 
        enc_input_names, &enc_input_tensor, 1, 
        enc_output_names, 1
    );

    // --- [新增] 打印 Audio Features 维度 ---
    auto audio_info = audio_features[0].GetTensorTypeAndShapeInfo();
    auto audio_shape = audio_info.GetShape();
    std::cout << "🔍 Audio Features Shape: [";
    for (size_t i = 0; i < audio_shape.size(); ++i) {
        std::cout << audio_shape[i] << (i < audio_shape.size() - 1 ? " x " : "");
    }
    std::cout << "]" << std::endl;
    // -------------------------------------

    // ==========================================
    // 5. 初始化 Decoder 循环
    // ==========================================
    std::vector<int64_t> generated_tokens;
    generated_tokens.push_back(50258); // <|startoftranscript|>
    
    const char* dec_input_names[] = {"tokens", "audio_features"};
    const char* dec_output_names[] = {"logits"};

    // 准备输入 Vector (连续内存布局)
    std::vector<Ort::Value> input_tensors;
    input_tensors.push_back(Ort::Value(nullptr));          // Index 0: 占位 (Tokens)
    input_tensors.push_back(std::move(audio_features[0])); // Index 1: Audio Features (移动进来)

    std::cout << "🗣️  Start Decoding..." << std::endl;
    std::cout << "--------------------------------" << std::endl;

    // ==========================================
    // 6. 主循环 (Main Loop)
    // ==========================================
    int max_steps = 100; // 防止死循环
    for (int step = 0; step < max_steps; step++) {
        size_t current_seq_len = generated_tokens.size();
        
        // 创建 Tokens Tensor
        const std::vector<int64_t> token_input_shape{1, static_cast<int64_t>(current_seq_len)};
        
        Ort::Value token_tensor = Ort::Value::CreateTensor(
            memory_info, 
            generated_tokens.data(), 
            current_seq_len, 
            token_input_shape.data(), 
            token_input_shape.size() // 必须是 2 (Rank)
        );

        // 更新 Input[0]
        input_tensors[0] = std::move(token_tensor);

        // 运行 Decoder
        std::vector<Ort::Value> dec_results = decoder_session->Run(
            Ort::RunOptions{nullptr}, 
            dec_input_names, 
            input_tensors.data(), // 传入 vector.data()
            2, 
            dec_output_names, 
            1
        );

        // 获取 Logits
        float* output_data = dec_results[0].GetTensorMutableData<float>();
        auto shape_info = dec_results[0].GetTensorTypeAndShapeInfo().GetShape();
        int64_t vocab_size = shape_info[2]; 
        int64_t seq_len = shape_info[1];    
        
        // 定位到最后一个时间步
        float* last_logits = output_data + (seq_len - 1) * vocab_size;
        
        // Greedy Search (Argmax)
        int next_token_id = 0;
        float max_val = -std::numeric_limits<float>::infinity();

        for (int i = 0; i < vocab_size; i++) {
            if (last_logits[i] > max_val) {
                max_val = last_logits[i];
                next_token_id = i;
            }
        }

        // 保存结果
        generated_tokens.push_back(next_token_id);
        
        // 解码并打印 (流式输出)
        std::string next_token_str = tokenizer.decode(next_token_id);
        std::cout << next_token_str << std::flush;

        // 结束判断
        if (next_token_id == 50257) { // <|endoftext|>
            std::cout << "\n--------------------------------" << std::endl;
            std::cout << "🛑 Reached <EOT>" << std::endl;
            break;
        }
    }

    return 0;
}
