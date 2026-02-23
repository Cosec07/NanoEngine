#include "tensor.hpp"
#include "ops.hpp"
#include <iostream>
#include "loader.hpp"
#include "model.hpp"
#include "tokenizer.hpp"
#include "safetensors.hpp"

int main() {
    try {
        std::cout<<"Starting NanoEngine Phase 2 Verification......"<<std::endl;

        Tokenizer tokenizer;
        tokenizer.load_json("vocab.json");

        Config config;
        Transformer model(config);

        SafeTensorsLoader loader("model.safetensors");
        model.load_weights(loader);

        int test_token_id = 310;
        std::string word = tokenizer.decode(test_token_id);

        std::cout<<"\nTarget Token ID: "<<test_token_id <<"-> Word: ["<< word <<"]"<<std::endl;

        Tensor hidden_state({(size_t)config.dim});

        std::cout << "Extracting embedding vector"<<std::endl;
        ops::get_embedding(model.token_embedding_table, test_token_id, hidden_state);

        std::cout<<"\n Success! First 5 values of the embedding vector: ["<< word << "]"<<std::endl;
        for(int i=0;i<5;++i) {
            std::cout<<"Dim["<<i<<"]"<<hidden_state[i]<<std::endl;
        }
    }catch(const std::exception& e) {
        std::cerr<<"Fatal Error: "<<e.what()<<std::endl;
        return 1;
    }
    return 0;
}