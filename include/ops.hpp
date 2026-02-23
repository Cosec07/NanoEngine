#ifndef OPS_HPP
#define OPS_HPP

#include "tensor.hpp"

namespace ops {
    void add(const Tensor& a, const Tensor& b, Tensor& out);

    void mul(const Tensor& a, const Tensor& b, Tensor& out);

    void relu(const Tensor& input, Tensor& output);

    float dot(const float* a, const float* b, size_t n);

    void matmul(const Tensor& a, const Tensor& b, Tensor& c);

    void softmax(Tensor& t);

    void layernorm(const Tensor& in, Tensor& out, float eps = 1e-5f);

    void get_embedding(const Tensor& embedding_table, int token_id, Tensor& out);
}

#endif