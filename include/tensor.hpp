#ifndef TENSOR_HPP
#define TENSOR_HPP

#include <vector>
#include <initializer_list>
#include <cassert>
#include <iostream>

class Tensor {
public:
    // Dimensions of the tensor (e.g., {1, 512, 4096})
    std::vector<size_t> shape;
    // The actual raw data
    std::vector<float> data;
    std::vector<float> strides;

    // Constructors
    Tensor() = default;
    Tensor(std::initializer_list<size_t> dims);
    Tensor(const std::vector<size_t>& dims);

    // Get total number of elements
    size_t size() const;

    // Basic indexing (flat)
    float& operator[](size_t index) { return data[index]; }
    const float& operator[](size_t index) const { return data[index]; }

    void compute_strides();

    float& operator()(std::initializer_list<size_t> indices);
    const float& operator()(std::initializer_list<size_t> indices) const;

    void print_info() const;
};

#endif