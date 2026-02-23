#include "tensor.hpp"
#include <numeric>

size_t Tensor::size() const {
    if (shape.empty()) return 0;
    size_t s=1;
    for(auto d: shape) s *=d;
    return s;
}

void Tensor::print_info() const {
    std::cout <<"Tensor Shape: [";
    for(size_t i=0; i<shape.size(); ++i) {
        std::cout<<shape[i]<<(i==shape.size() - 1 ? "" : ", ");
    }
    std::cout <<"] | Total Elements:" << size() << std::endl;
}

Tensor::Tensor(std::initializer_list<size_t> dims) : shape(dims) {
    compute_strides();
    data.resize(size(), 0.0f);
}

void Tensor::compute_strides() {
    strides.resize(shape.size());
    size_t stride = 1;
    for(int i = shape.size() - 1; i>=0; --i) {
        strides[i] = stride;
        stride*= shape[i];
    }
}

float& Tensor::operator()(std::initializer_list<size_t> indices) {
    assert(indices.size() == shape.size());
    size_t offset = 0;
    auto it = indices.begin();
    for(size_t i=0; i < indices.size(); ++i) {
        offset += (*it) * strides[i];
        it++;
    }
    return data[offset];
}

const float& Tensor::operator()(std::initializer_list<size_t> indices) const {
    assert(indices.size() == shape.size());
    size_t offset = 0;
    auto it = indices.begin();
    for(size_t i=0; i < indices.size(); ++i) {
        offset += (*it) * strides[i];
        it++;
    }
    return data[offset];
}