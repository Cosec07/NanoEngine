#include "safetensors.hpp"
#include <../third_party/json.hpp>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <cstring>

using json = nlohmann::json;

float SafeTensorsLoader::bf16_to_fp32(uint16_t b) const {
    uint32_t val = static_cast<uint32_t>(b) << 16;
    float f;
    std::memcpy(&f, &val, sizeof(f));
    return f;
}

SafeTensorsLoader::SafeTensorsLoader(const std::string& path) : file_path(path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) throw std::runtime_error("Cannot open safetensors file: " + path);

    uint64_t header_size;
    file.read(reinterpret_cast<char*>(&header_size), sizeof(header_size));

    std::string header_json(header_size, ' ');
    file.read(&header_json[0], header_size);

    raw_data_start = 8 + header_size;

    json metadata = json::parse(header_json);

    for (auto& [key,val] : metadata.items()) {
        if ( key == "__metadata__") continue;

        TensorInfo info;
        info.dtype = val["dtype"];
        info.start_offset = val["data_offsets"][0];
        info.end_offset = val["data_offsets"][1];
        tensor_map[key] = info;
    }

    std::cout<<"SafeTensors parsed: Found"<< tensor_map.size()<<"tensors." <<std::endl;
}

bool SafeTensorsLoader::contains(const std::string& tensor_name) const {
    return tensor_map.find(tensor_name) != tensor_map.end();
}

void SafeTensorsLoader::load_tensor(const std::string& tensor_name, Tensor& out_tensor) const {
    if (tensor_map.find(tensor_name) == tensor_map.end()) {
        throw std::runtime_error("Tensor not found in safetensors file: " + tensor_name);
    }

    const TensorInfo& info = tensor_map.at(tensor_name);
    std::ifstream file(file_path, std::ios::binary);

    file.seekg(raw_data_start + info.start_offset, std::ios::beg);

    if(info.dtype == "BF16") {
        size_t num_elements = (info.end_offset - info.start_offset) / 2;
        if (num_elements != out_tensor.size()) {
            throw std::runtime_error("Tensor size mismatch for " + tensor_name);
        }

        std::vector<uint16_t> buffer(num_elements);
        file.read(reinterpret_cast<char*>(buffer.data()),num_elements * sizeof(uint16_t));

        for(size_t i=0; i<num_elements;++i) {
            out_tensor.data[i] = bf16_to_fp32(buffer[i]);
        }

    }else {
        throw std::runtime_error("Unsupported dtype: " + info.dtype);
    }

}