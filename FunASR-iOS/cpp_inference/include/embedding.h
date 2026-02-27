#pragma once

#include <vector>
#include <string>
#include <fstream>
#include <iostream>

class EmbeddingLayer {
public:
    std::vector<float> weights;
    int vocab_size = 51865;
    int hidden_size = 768; // Check if your model is Tiny (384) or Base (512) or Small (768)

    bool load_weights(const std::string& path) {
        std::ifstream file(path, std::ios::binary);
        if (!file.is_open()) {
            std::cerr << "Failed to open embedding weights: " << path << std::endl;
            return false;
        }
        
        // Allocate memory
        weights.resize(vocab_size * hidden_size);
        
        // Read binary data
        file.read(reinterpret_cast<char*>(weights.data()), weights.size() * sizeof(float));
        return true;
    }

    // CPU Lookup: Convert Token IDs (Int) to Embeddings (Float)
    std::vector<float> forward(const std::vector<int64_t>& input_ids) {
        std::vector<float> output;
        output.reserve(input_ids.size() * hidden_size);

        for (int64_t id : input_ids) {
            if (id < 0 || id >= vocab_size) {
                // Handle out of bounds (zero vector or error)
                output.insert(output.end(), hidden_size, 0.0f);
                continue;
            }
            
            // Copy the corresponding vector
            const float* src = weights.data() + (id * hidden_size);
            output.insert(output.end(), src, src + hidden_size);
        }
        return output;
    }
};