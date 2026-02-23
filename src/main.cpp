#include "model.hpp"
#include "safetensors.hpp"
#include "tokenizer.hpp"
#include "ops.hpp"
#include <iostream>
#include <vector>

int main() {
    try {
        std::cout << "=== Nano-Engine Live ===" << std::endl;

        // 1. Setup
        Tokenizer tokenizer;
        tokenizer.load_json("vocab.json");

        Config config; 
        config.max_seq_len = 1024; // Context window size
        Transformer model(config);

        SafeTensorsLoader loader("model.safetensors");
        model.load_weights(loader);

        std::cout << "\n----------------------------------------\n";

        int next_token = 400; 
        
        std::cout << tokenizer.decode(next_token) << std::flush;

        int max_tokens_to_generate = 10; 
        
        for (int pos = 0; pos < max_tokens_to_generate; ++pos) {
            

            Tensor logits = model.forward(next_token, pos, config);

 
            next_token = ops::argmax(logits);


            std::string word = tokenizer.decode(next_token);

            std::cout << word << std::flush;

            if (next_token == 151645 || next_token == 151643) {
                break;
            }
        }
        
        std::cout << "\n\n[Generation Complete]" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "\nFatal Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}