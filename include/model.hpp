#ifndef MODEL_HPP
#define MODEL_HPP

#include "tensor.hpp"
#include "safetensors.hpp"
#include <vector>
#include <optional>

struct Config{
    int dim = 1024;
    int hidden_dim = 3072;
    int n_layers = 28;
    int n_heads = 16;
    int n_kv_heads = 8;
    int head_dim = 128;
    int vocab_size = 151936;
    int max_seq_len = 40960;
    bool tie_embeddings = true;
    float rms_norm_eps = 1e-6f;
};

struct TransformerLayer {
    Tensor rms_att_weight;

    Tensor wq; 
    Tensor wk; 
    Tensor wv;
    Tensor wo;

    Tensor rms_ffn_weight;

    Tensor w_gate;
    Tensor w_up;
    Tensor w_down;

    Tensor key_cache;
    Tensor value_cache;

    void forward(Tensor& hidden_state, int pos, const Config& config);
};

struct Transformer{
    Config config;

    Tensor token_embedding_table;

    std::vector<TransformerLayer> layers;

    Tensor rms_final_weight;

    Tensor w_cls;
    
    Transformer(Config cfg);

    void load_weights(const SafeTensorsLoader& loader);

    Tensor forward(int token_id, int pos, Config& config);
};
#endif