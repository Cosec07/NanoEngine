#include "ops.hpp"
#include <algorithm>
#include <numeric>
#include <cmath>
#include <cstring>

namespace ops {

    void add(const Tensor& a, const Tensor& b, Tensor& out) {
        assert(a.size() == b.size() && a.size() == out.size());
        for(size_t i=0; i< a.size(); ++i) {
            out[i] = a[i] + b[i];
        }
    }

    void mul(const Tensor& a, const Tensor& b, Tensor& out) {
        assert(a.size() == b.size() && a.size() == out.size());
        for(size_t i=0; i< a.size(); ++i) {
            out[i] = a[i] * b[i];
        }
    }

    void relu(const Tensor& inp, Tensor& out) {
        assert(inp.size() == out.size());
        for(size_t i=0; i< inp.size(); ++i) {
            out[i] = std::max(0.0f, inp[i]);
        }
    }

    float dot(const float* a, const float*  b, size_t n) {
        float sum=0.0f;
        for(size_t i=0; i < n; ++i) {
            sum += a[i] * b[i];
        }
        return sum;
    }

    void matmul(const Tensor& a, const Tensor& b, Tensor& c) {
    size_t M = a.shape[0];
    size_t K = a.shape[1];
    size_t N = b.shape[1];

    assert(a.shape[1] == b.shape[0]); // Inner dimensions must match
    assert(c.shape[0] == M && c.shape[1] == N);

    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (size_t k = 0; k < K; ++k) {
                // Accessing: A[i, k] * B[k, j]
                sum += a({i, k}) * b({k, j});
            }
            c({i, j}) = sum;
        }
        }
    }

    void softmax(Tensor& t) {
        float max_val = t[0];
        for(size_t i=1; i<t.size(); ++i) if(t[i] > max_val) max_val = t[i];
        
        float sum = 0.0f;
        for(size_t i=0; i < t.size(); ++i) {
            t[i] = std::exp(t[i] - max_val);
            sum +=t[i];
        }

        for(size_t i=0; i<t.size(); ++i) {
            t[i] /=sum;
        }
    }

    void layernorm(const Tensor& in, Tensor& out, float eps) {
        float mean=0.0f;
        for(size_t i=0; i<in.size(); ++i) mean += in[i];
        mean /= in.size();

        float var=0.0f;
        for(size_t i=0; i<in.size(); ++i) {
            float diff = in[i] - mean;
            var += diff * diff;
        }
        var /= in.size();

        for(size_t i=0; i<in.size(); ++i) {
            out[i] = (in[i] - mean) / std::sqrt(var + eps);
        }
    }

    void get_embedding(const Tensor& embedding_table, int token_id, Tensor& out) {
        size_t dim = out.size();

        assert(embedding_table.shape[1] == dim);

        size_t offset = static_cast<size_t>(token_id) * dim;

        std::memcpy(out.data.data(), embedding_table.data.data() + offset, dim * sizeof(float));
    }
}