#ifndef SAFETENSORS_HPP
#define SAFETENSORS_HPP

#include "tensor.hpp"
#include <string>
#include <map>
#include <cstdint>

struct TensorInfo {
    size_t start_offset;
    size_t end_offset;
    std::string dtype;
};

class SafeTensorsLoader{
private:
    std::string file_path;
    size_t raw_data_start;
    std::map<std::string,  TensorInfo> tensor_map;

    float bf16_to_fp32(uint16_t b) const;
public:
    SafeTensorsLoader(const std::string& path);

    void load_tensor(const std::string& tensor_name, Tensor& out_tensor) const;
};

#endif