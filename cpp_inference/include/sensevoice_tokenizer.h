#pragma once

#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <iostream>

class SenseVoiceTokenizer {
public:
    bool load(const std::string& tokens_path) {
        std::ifstream file(tokens_path);
        if (!file.is_open()) {
            std::cerr << "Failed to open tokens file: " << tokens_path << std::endl;
            return false;
        }

        std::string line;
        int max_id = -1;

        // First pass: find max token ID
        std::vector<std::pair<std::string, int>> entries;
        while (std::getline(file, line)) {
            if (line.empty()) continue;
            // Format: "token_string token_id"
            auto last_space = line.rfind(' ');
            if (last_space == std::string::npos) continue;

            std::string token = line.substr(0, last_space);
            int id = std::stoi(line.substr(last_space + 1));
            entries.push_back({token, id});
            if (id > max_id) max_id = id;
        }

        // Build lookup table
        vocab_.resize(max_id + 1);
        for (auto& [token, id] : entries) {
            vocab_[id] = token;
        }

        std::cout << "Loaded " << entries.size() << " tokens (max_id=" << max_id << ")" << std::endl;
        return true;
    }

    std::string decode(int token_id) const {
        if (token_id < 0 || token_id >= (int)vocab_.size()) {
            return "";
        }
        const std::string& tok = vocab_[token_id];
        // SentencePiece uses ▁ (U+2581) as word boundary marker, replace with space
        std::string result;
        result.reserve(tok.size());
        size_t i = 0;
        while (i < tok.size()) {
            // UTF-8 for ▁ is 0xE2 0x96 0x81
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
        return result;
    }

    std::string decode(const std::vector<int>& token_ids) const {
        std::string result;
        for (int id : token_ids) {
            result += decode(id);
        }
        return result;
    }

private:
    std::vector<std::string> vocab_;
};
