#ifndef LOADER_HPP
#define LOADER_HPP

#include "tensor.hpp"
#include <string>

namespace loader {
    void load_raw(const std::string& path, Tensor& t);
}

#endif