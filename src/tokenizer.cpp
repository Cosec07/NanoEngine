#include "tokenizer.hpp"
#include "../third_party/json.hpp"
#include <fstream>
#include <iostream>
#include <stdexcept>

using json = nlohmann::json;

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
    }
    std::cout<<"Loaded "<<vocab.size()<<"token from"<<path<<std::endl; 
}

std::string Tokenizer::decode(int token_id) const {
    if (token_id < 0 || token_id >= vocab.size()) {
        return "<UNK>";
    }
    return vocab[token_id]; 
}