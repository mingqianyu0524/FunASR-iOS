#include <fstream>
#include <iostream>
#include "json.hpp"
#include "whisper_tokenizer.h"

using json = nlohmann::json;

WhisperTokenizer::WhisperTokenizer() {}

bool WhisperTokenizer::load(const std::string& vocab_path)
{
    std::ifstream f(vocab_path);
    if (!f.is_open()) {
        std::cerr << "Failed to open vocab file" << std::endl;
        return false;
    }
    try {
        json data = json::parse(f);
        for (auto& [key, value] : data.items()) {
            this->vocab_[std::stoi(key)] = value.get<std::string>();
        }
    } catch (const json::parse_error& e) {
        std::cerr << "Failed to parse json, error = " << e.what() << std::endl;
        return false;
    }
    return true;
}

std::string WhisperTokenizer::decode(int token_id) const
{
    auto it = vocab_.find(token_id);
    if (it == vocab_.end()) {
        return "<UNK>";
    }
    return it->second;
}

std::string WhisperTokenizer::decode(const std::vector<int>& token_ids) const
{
    std::string ret;
    ret.reserve(token_ids.size() * 4); 
    for (int token_id : token_ids) {
        auto it = vocab_.find(token_id);
        if (it == vocab_.end()) {
            ret += "<UNK>";
        } else {
            ret += it->second;
        }
    }
    return ret;
}

bool WhisperTokenizer::is_eot(int token_id) const
{
    return token_id == EOT_TOKEN_ID;
}
