#include "tokenizer.hpp"
#include "../third_party/json.hpp"
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <sstream>

using json = nlohmann::json;

void replace_all(std::string& str, const std::string& from, const std::string& to) {
    if (from.empty()) return;

    size_t start_pos = 0;
    while((start_pos = str.find(from, start_pos)) != std::string::npos) {
        str.replace(start_pos, from.length(), to);
        start_pos += to.length();
    }
}

void Tokenizer::load_json(const std::string& path) {
    std::ifstream file(path);
    if(!file.is_open()) {
        throw std::runtime_error("Could not open" + path);
    }
    std::cout<<"Parsing JSON..."<<std::endl;
    json vocab_data = json::parse(file);

    int max_id = 0;
    for(const auto& item : vocab_data.items()) {
        int id = item.value().get<int>();
        if (id > max_id) {
            max_id = id;
        }
    }

    vocab.resize(max_id + 1, "<UNK>");

    for(const auto& item : vocab_data.items()) {
        int id = item.value().get<int>();
        std::string token_str = item.key();
        vocab[id] = token_str;
        string_to_id[token_str] = id;
    }
    std::cout<<"Loaded "<<vocab.size()<<"token from"<<path<<std::endl; 
}

std::string Tokenizer::decode(int token_id) const {
    if (token_id < 0 || token_id >= (int)vocab.size()) {
        return "";
    }
    
    std::string word = vocab[token_id];

    replace_all(word, "Ġ", " ");
    replace_all(word, "Ċ", "\n");

    return word; 
}

std::vector<int> Tokenizer::encode(const std::string& text) const {
    std::string processed = text;
    replace_all(processed, " ", "Ġ");
    replace_all(processed, "\n", "Ċ");

    std::vector<int> tokens;
    for (size_t i = 0; i < processed.length(); ) {
        int best_id = -1;
        size_t best_len = 0;

        // Greedy search for the longest token
        // Use a reasonable lookahead (e.g., 64 characters)
        size_t max_lookahead = std::min((size_t)64, processed.length() - i);
        for (size_t len = max_lookahead; len >= 1; --len) {
            std::string sub = processed.substr(i, len);
            auto it = string_to_id.find(sub);
            if (it != string_to_id.end()) {
                best_id = it->second;
                best_len = len;
                break;
            }
        }

        if (best_id != -1) {
            tokens.push_back(best_id);
            i += best_len;
        } else {
            // Fallback for characters not in vocab
            i++;
        }
    }
    return tokens;
}

