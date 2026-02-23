#include "model.hpp"
#include <iostream>

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

        layers.push_back(layer);

    }

    rms_final_weight = Tensor({(size_t)cfg.dim});

    if(cfg.tie_embeddings) {
        std::cout <<"Weight Tying Enabled: Output head shares memory with embeddings layer" << std::endl;
    }else {
        w_cls = Tensor({(size_t)cfg.vocab_size, (size_t)cfg.dim});
    }
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

    if(config.tie_embeddings) {
        std::cout << "Skipping lm_head.weight (Weights are tied to embeddings)."<<std::endl;
    }else {
        loader.load_tensor("model.lm_head.weight", w_cls);
    }
    std::cout<<"All weights loaded successfully!"<< std::endl;
}