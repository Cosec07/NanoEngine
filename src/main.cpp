#include "model.hpp"
#include "safetensors.hpp"
#include "tokenizer.hpp"
#include "ops.hpp"
#include <iostream>
#include <vector>
#include <random>
#include <string>
#include <chrono>
#include <iomanip>

struct ChatMessage {
    std::string role;
    std::string content;
};

class ChatML {
public:
    static std::vector<int> format(const Tokenizer& tokenizer, const std::vector<ChatMessage>& messages) {
        std::vector<int> tokens;
        for (const auto& msg : messages) {
            tokens.push_back(151644); // <|im_start|>
            std::vector<int> body = tokenizer.encode(msg.role + "\n" + msg.content);
            tokens.insert(tokens.end(), body.begin(), body.end());
            tokens.push_back(151645); // <|im_end|>
            tokens.push_back(198);    // \n
        }
        // Append Assistant trigger for the model to start generating
        tokens.push_back(151644); // <|im_start|>
        std::vector<int> header = tokenizer.encode("assistant\n");
        tokens.insert(tokens.end(), header.begin(), header.end());
        
        return tokens;
    }
};

int main() {
    try {
        std::cout << "=== Nano-Engine: Qwen 3 0.6B ===\n" << std::endl;

        
        // 1. Setup Tokenizer and Model
        Tokenizer tokenizer;
        tokenizer.load_json("vocab.json");

        Config config; 
        config.max_seq_len = 2048; // Standard testing context length
        Transformer model(config);

        SafeTensorsLoader loader("model.safetensors");
        model.load_weights(loader);

        // Initialize Random Number Generator for sampling
        std::random_device rd;
        std::mt19937 rng(rd());
        float temperature = 0.7f; // Recommended for general chat
        float top_p = 0.8f;       // Standard Top-P value

        int pos = 0;

        std::cout << "User: ";
        std::string user_input;
        std::getline(std::cin, user_input);

        // 2. Build ChatML Prompt
        std::vector<ChatMessage> history = {
            {"system", "You are a helpful AI assistant."},
            {"user", user_input}
        };
        
        std::vector<int> prompt_tokens = ChatML::format(tokenizer, history);

        std::cout << "DEBUG: Prompt Tokens: ";
        for(auto t : prompt_tokens) std::cout << t << " ";
        std::cout << std::endl;

        std::cout << "Nano: " << std::flush;
        auto start_time = std::chrono::high_resolution_clock::now();
        int generated_tokens = 0;
        // 3. Prefill Phase
        Tensor logits({(size_t)config.vocab_size});
        for (int token : prompt_tokens) {
            logits = model.forward(token, pos, config);
            pos++;
        }

        // 4. Generation Phase
        int max_tokens = 512;
        for (int i = 0; i < max_tokens; ++i) {
            generated_tokens++;
            // Apply Temperature and Sample
            for (size_t j = 0; j < logits.size(); ++j) logits[j] /= temperature;
            ops::softmax(logits);
            int next_token = ops::sample_topp(logits, top_p, rng);

            // Stop if EOS or ChatML end token is reached
            if (next_token == 151645 || next_token == 151643) break;

            // Decode and print if not filler
            if (next_token != 151644) {
                std::string piece = tokenizer.decode(next_token);
                // Further filter blank lines if the piece is just a newline and it's the very start
                if (!(i == 0 && (piece == "\n" || piece == "\r\n"))) {
                    std::cout << piece << std::flush;
                }
            }

            // ALWAYS forward pass for next token
            logits = model.forward(next_token, pos, config);
            pos++;

            if (pos >= config.max_seq_len) break;
        }
        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = end_time - start_time;
        double seconds = duration.count();
        double tps = generated_tokens / seconds;
        std::cout << "\n\n========================================" << std::endl;
        std::cout << "[Generation Complete]" << std::endl;
        std::cout << "Tokens Generated: " << generated_tokens << std::endl;
        std::cout << "Time Elapsed:     " << seconds << " seconds" << std::endl;
        std::cout << "Speed:            " << std::fixed << std::setprecision(2) << tps << " tokens/sec" << std::endl;
        std::cout << "========================================" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Fatal: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}