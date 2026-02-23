#ifndef TOKENIZER_HPP
#define TOKENIZER_HPP

#include <string>
#include <vector>

class Tokenizer {
public:
    std::vector<std::string> vocab;
    Tokenizer() = default;

    void load_json(const std::string& path);

    std::string decode(int token_id) const;
};

#endif