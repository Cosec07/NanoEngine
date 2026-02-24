#ifndef OPS_HPP
#define OPS_HPP

#include "tensor.hpp"
#include <random>

namespace ops {
    void add(const Tensor& a, const Tensor& b, Tensor& out);

    void mul(const Tensor& a, const Tensor& b, Tensor& out);

    void relu(const Tensor& input, Tensor& output);

    float dot(const float* a, const float* b, size_t n);

    void matmul(const Tensor& a, const Tensor& b, Tensor& c);

    void softmax(Tensor& t);

    void rmsnorm(const Tensor& in, Tensor& out, float eps = 1e-6f);

    void get_embedding(const Tensor& embedding_table, int token_id, Tensor& out);

    void matvec(const Tensor& m, const Tensor& v, Tensor& out);

    void apply_rope(Tensor& t, int pos, int head_dim, float rope_theta = 1000000.0f);

    void silu(Tensor& t);

    int argmax(const Tensor& logits);

    int sample(Tensor& probs, float top_p, int top_k, std::mt19937& rng);

    int sample_topp(Tensor& probs, float top_p, std::mt19937& rng);
}

#endif