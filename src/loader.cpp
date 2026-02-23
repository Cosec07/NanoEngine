#include "loader.hpp"
#include <fstream>
#include <iostream>
#include <stdexcept>


    void loader::load_raw(const std::string& path, Tensor& t) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);

    if(!file){
        throw std::runtime_error("Could not open file:" + path);
    }

    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

   size_t expected_bytes = t.size() * sizeof(float);

   if(size != expected_bytes) {
    std::cerr << "Warning: File size ("<< size << ") does not match Tensor size ("<< expected_bytes <<")"<<std::endl;
   }

   if(file.read(reinterpret_cast<char*>(t.data.data()), expected_bytes)) {
    std::cout<<"Successfully loaded"<<expected_bytes<<"bytes from"<<path<<std::endl;
   }
   else{
    throw std::runtime_error("Error reading file");
   }
}

