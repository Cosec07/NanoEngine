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

    void rmsnorm(const Tensor& in, Tensor& out, float eps) {
        float ss = 0.0f;
        for (size_t i = 0; i < in.size(); ++i) {
            ss += in[i] * in[i];
        }
        ss /= in.size();
        ss += eps;
        ss = 1.0f / std::sqrt(ss);
        for (size_t i = 0; i < in.size(); ++i) {
            out[i] = in[i] * ss;
        }
    }

    void get_embedding(const Tensor& embedding_table, int token_id, Tensor& out) {
        size_t dim = out.size();

        assert(embedding_table.shape[1] == dim);

        size_t offset = static_cast<size_t>(token_id) * dim;

        std::memcpy(out.data.data(), embedding_table.data.data() + offset, dim * sizeof(float));
    }

    void matvec(const Tensor& m, const Tensor& v, Tensor& out) {
        size_t rows = m.shape[0];
        size_t cols = m.shape[1];

        assert(v.shape.size() == 1 || (v.shape.size() == 2 && v.shape[0] == 1));
        assert(v.size() == cols);
        assert(out.size() == rows);

        for (size_t i=0; i<rows; ++i) {
            out[i] = dot(m.data.data() + (i * cols), v.data.data(), cols);
        }
    }

    void apply_rope(Tensor& t, int pos, int head_dim, float rope_theta) {
        size_t n_heads = t.size() / head_dim;
        int half_dim = head_dim / 2;

        for(size_t h=0; h<n_heads; ++h) {
            float* head_base = t.data.data() + (h * head_dim);

            for (size_t i=0; i<half_dim; ++i) {
                float freq = 1.0f / std::pow(rope_theta, 2.0f * i / head_dim);
                
                float val = static_cast<float>(pos) * freq;
                float fcr = std::cos(val);
                float fci = std::sin(val);

                float v0 = head_base[i];
                float v1 = head_base[i + half_dim];

                head_base[i] = v0 * fcr - v1 * fci;
                head_base[i + half_dim] = v0 * fci + v1 * fcr;
            }
        }

    }

    void silu(Tensor& t) {
        for(size_t i=0; i< t.size(); ++i) {
            float x = t[i];
            t[i] = x / (1.0f + std::exp(-x));
        }
    }

    int argmax(const Tensor& logits) {
        int max_index = 0;
        float max_value = logits[0];
        for(size_t i=1; i<logits.size();++i){
            if(logits[i] > max_value) {
                max_value = logits[i];
                max_index = i;
            }
        }
        return max_index;
    }

    struct ProbIndex {
        float prob;
        int index;
    };

    int sample(Tensor& probs, float top_p, int top_k, std::mt19937& rng) {
    size_t n = probs.size();
    std::vector<ProbIndex> sorted_probs(n);
    for (size_t i = 0; i < n; ++i) {
        sorted_probs[i] = {probs[i], static_cast<int>(i)};
    }

    // 1. Sort in descending order based on probability
    std::sort(sorted_probs.begin(), sorted_probs.end(), 
        [](const ProbIndex& a, const ProbIndex& b) {
            return a.prob > b.prob;
        });

    // 2. Apply Top-K cutoff
    // If top_k is 20, we only look at the first 20 elements.
    size_t limit = n;
    if (top_k > 0 && top_k < n) {
        limit = top_k;
    }

    // 3. Apply Top-P cutoff on the remaining items
    float cumulative_prob = 0.0f;
    size_t cutoff_index = 0;
    for (size_t i = 0; i < limit; ++i) {
        cumulative_prob += sorted_probs[i].prob;
        cutoff_index = i;
        if (cumulative_prob >= top_p) {
            break;
        }
    }

    // 4. Sample randomly from the final restricted pool
    std::uniform_real_distribution<float> dist(0.0f, cumulative_prob);
    float r = dist(rng);
    
    float current_sum = 0.0f;
    for (size_t i = 0; i <= cutoff_index; ++i) {
        current_sum += sorted_probs[i].prob;
        if (r <= current_sum) {
            return sorted_probs[i].index;
        }
    }
    
    return sorted_probs[cutoff_index].index; 
}

    int sample_topp(Tensor& probs, float top_p, std::mt19937& rng) {
        return sample(probs, top_p, 0, rng);
    }

}
