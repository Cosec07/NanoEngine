#include "model.hpp"
#include <iostream>
#include "ops.hpp"
#include <cmath>
#include <cstring>

Transformer::Transformer(Config cfg) : config(cfg) {
    size_t q_dim = cfg.n_heads * cfg.head_dim; // 16(heads) * 128 = 2048
    size_t kv_dim = cfg.head_dim * cfg.n_kv_heads; // 64(head dim) * 8(kv heads) = 512

    std::cout <<"Model Config:  " << cfg.dim << "dim, "
                << cfg.n_layers << "layers, "
                << cfg.n_heads << "heads, "
                << cfg.n_kv_heads << "kv heads, "
                << cfg.vocab_size << "vocab size, "
                << cfg.max_seq_len << "max seq len" << std::endl;
    std::cout <<"GQA Reduction: Key/Value tensors are "
                << (float)kv_dim / cfg.dim * 100.0f << "% size of the Query tensors"<< std::endl;

    //1. Embeddings
    token_embedding_table = Tensor({(size_t)cfg.vocab_size, (size_t)cfg.dim});

    //2. Layers
    layers.reserve(cfg.n_layers);
    for(int i=0; i<cfg.n_layers; ++i) {
        TransformerLayer layer;

        layer.rms_att_weight = Tensor({(size_t)cfg.dim});

        //WQ: (1204, 1024)
        layer.wq = Tensor({q_dim, (size_t)cfg.dim});

        //WK/WV: (512, 1024)
        layer.wk = Tensor({kv_dim, (size_t)cfg.dim});
        layer.wv = Tensor({kv_dim, (size_t)cfg.dim});

        // WO: (1024, 1024)
        layer.wo = Tensor({(size_t)cfg.dim, q_dim});

        layer.rms_ffn_weight = Tensor({(size_t)cfg.dim});

        layer.w_gate = Tensor({(size_t)cfg.hidden_dim, (size_t)cfg.dim});
        layer.w_up = Tensor({(size_t)cfg.hidden_dim, (size_t)cfg.dim});
        layer.w_down = Tensor({(size_t)cfg.dim, (size_t)cfg.hidden_dim});

        layer.key_cache = Tensor({(size_t)cfg.max_seq_len, kv_dim});
        layer.value_cache = Tensor({(size_t)cfg.max_seq_len, kv_dim});

        layers.push_back(layer);

    }

    rms_final_weight = Tensor({(size_t)cfg.dim});

    // Always allocate w_cls to support loading it if present
    w_cls = Tensor({(size_t)cfg.vocab_size, (size_t)cfg.dim});
}

void Transformer::load_weights(const SafeTensorsLoader& loader) {
    std::cout<<"Loading weights into memory....."<<std::endl;

    loader.load_tensor("model.embed_tokens.weight", token_embedding_table);

    for(int i=0; i<config.n_layers;++i){
        std::string prefix = "model.layers." + std::to_string(i) + ".";

        //Attention norm
        loader.load_tensor(prefix + "input_layernorm.weight", layers[i].rms_att_weight);

        //Attention Projections
        loader.load_tensor(prefix + "self_attn.q_proj.weight", layers[i].wq);

        loader.load_tensor(prefix + "self_attn.k_proj.weight", layers[i].wk);
        loader.load_tensor(prefix + "self_attn.v_proj.weight", layers[i].wv);
        loader.load_tensor(prefix + "self_attn.o_proj.weight", layers[i].wo);

        //FNN Norm
        loader.load_tensor(prefix + "post_attention_layernorm.weight", layers[i].rms_ffn_weight);

        //FFN (SwiGLU)
        loader.load_tensor(prefix + "mlp.gate_proj.weight", layers[i].w_gate);
        loader.load_tensor(prefix + "mlp.up_proj.weight", layers[i].w_up);
        loader.load_tensor(prefix + "mlp.down_proj.weight", layers[i].w_down);

        if (i % 4 == 0) {
            std::cout <<"Loaded layer "<< i<< "/"<<config.n_layers<<std::endl;
        }
    }
    
    //Final Norm
    loader.load_tensor("model.norm.weight", rms_final_weight);

    // Load lm_head if present
    if (loader.contains("model.lm_head.weight")) {
        std::cout << "Found model.lm_head.weight, loading..." << std::endl;
        loader.load_tensor("model.lm_head.weight", w_cls);
    } else if(config.tie_embeddings) {
        std::cout << "model.lm_head.weight not found. Using tied embeddings."<<std::endl;
        // Point w_cls data to embeddings (copy for simplicity, or just use flag in forward)
        // Since we have logic in forward, we don't need to copy if we rely on flag. 
        // BUT if we want to be safe, let's just copy.
        // Actually, let's rely on forward logic but we need to know if we loaded it.
        // We can check if w_cls was touched? 
        // Or better: update forward logic to check if loaded?
        // Simpler: Just copy data if tied and not found.
        std::memcpy(w_cls.data.data(), token_embedding_table.data.data(), token_embedding_table.size() * sizeof(float));
    } else {
        std::cerr << "Warning: lm_head not found and embeddings not tied!" << std::endl;
    }
    std::cout<<"All weights loaded successfully!"<< std::endl;
}

void TransformerLayer::forward(Tensor& hidden_state, int pos,const Config& config) {
    size_t q_dim = config.n_heads * config.head_dim;
    size_t kv_dim = config.n_kv_heads * config.head_dim;

    Tensor q({q_dim});
    Tensor k({kv_dim});
    Tensor v({kv_dim});
    Tensor normalized_hidden({(size_t)config.dim});

    // Input RMSNorm
    ops::rmsnorm(hidden_state, normalized_hidden, config.rms_norm_eps);
    ops::mul(normalized_hidden, rms_att_weight, normalized_hidden);

    ops::matvec(wq, normalized_hidden, q);
    ops::matvec(wk, normalized_hidden, k);
    ops::matvec(wv, normalized_hidden, v);

    ops::apply_rope(q, pos, config.head_dim, 1000000.0f);
    ops::apply_rope(k, pos, config.head_dim, 1000000.0f);

    size_t kv_offset = pos * kv_dim;
    std::memcpy(key_cache.data.data() + kv_offset, k.data.data(), kv_dim * sizeof(float));
    std::memcpy(value_cache.data.data() + kv_offset, v.data.data(), kv_dim * sizeof(float));

    Tensor att_out({q_dim});

    float scale  = 1.0f / std::sqrt(static_cast<float>(config.head_dim));

    int kv_mul = config.n_heads / config.n_kv_heads;

    for (int h=0; h< config.n_heads; ++h) {
        int kv_h = h / kv_mul;

        float* q_head = q.data.data() + (h * config.head_dim);
        float* out_head = att_out.data.data() + (h * config.head_dim);

        std::vector<float> scores(pos + 1, 0.0f);

        for(int t=0; t<=pos; ++t) {
            float* k_cache_head = key_cache.data.data() + (t * kv_dim) + (kv_h * config.head_dim);

            scores[t] = ops::dot(q_head, k_cache_head, config.head_dim) * scale;
        }
        float max_val = scores[0];
        for(int t = 1; t<=pos;++t) {
            if(scores[t] > max_val) max_val = scores[t];
        }
        float sum = 0.0f;
        for(int t=0; t<=pos;++t) {
            scores[t] = std::exp(scores[t] - max_val);
            sum += scores[t];
        }
        for (int t=0; t<=pos; ++t){
            scores[t] /=sum;
        }
        
        for(size_t i=0; i<config.head_dim; ++i) out_head[i] = 0.0f;
        for(int t=0; t<=pos; ++t) {
            float* v_cache_head = value_cache.data.data() + (t * kv_dim) + (kv_h * config.head_dim);
            float weight = scores[t];

            for(size_t i=0; i<config.head_dim; ++i) {
                out_head[i] += weight * v_cache_head[i];
            }
        }
    }

    Tensor projected_out({(size_t)config.dim});
    ops::matvec(wo, att_out, projected_out);

    ops::add(hidden_state, projected_out, hidden_state);

    Tensor ffn_norm({(size_t)config.dim});
    // Post-Attention RMSNorm
    ops::rmsnorm(hidden_state, ffn_norm, config.rms_norm_eps);
    ops::mul(ffn_norm, rms_ffn_weight, ffn_norm);

    Tensor gate_out({(size_t)config.hidden_dim});
    Tensor up_out({(size_t)config.hidden_dim});

    ops::matvec(w_gate, ffn_norm, gate_out);
    ops::matvec(w_up, ffn_norm, up_out);

    ops::silu(gate_out);

    ops::mul(gate_out, up_out, gate_out);

    Tensor down_out({(size_t)config.dim});
    ops::matvec(w_down, gate_out, down_out);

    ops::add(hidden_state, down_out, hidden_state);
}

Tensor Transformer::forward(int token_id, int pos, Config& config) {
    Tensor hidden_state({(size_t)config.dim});
    ops::get_embedding(token_embedding_table, token_id, hidden_state);

    for(int i=0; i<config.n_layers; ++i){
        layers[i].forward(hidden_state, pos, config);
    }

    Tensor final_norm({(size_t)config.dim});
    // Final RMSNorm
    ops::rmsnorm(hidden_state, final_norm, config.rms_norm_eps);
    ops::mul(final_norm, rms_final_weight, final_norm);

    Tensor logits({(size_t)config.vocab_size});

    // Always use w_cls (which is either loaded or copied from embeddings)
    ops::matvec(w_cls, final_norm, logits);

    return logits;
}

