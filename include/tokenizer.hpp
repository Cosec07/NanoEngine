#ifndef TOKENIZER_HPP
#define TOKENIZER_HPP

#include <string>
#include <vector>
#include <unordered_map>
class Tokenizer {
public:
    std::vector<std::string> vocab;
    std::unordered_map<std::string, int> string_to_id;

    Tokenizer() = default;
    void load_json(const std::string& path);
    std::string decode(int token_id) const;

    std::vector<int> encode(const std::string& text) const;
};

#endif