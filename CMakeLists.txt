cmake_minimum_required(VERSION 3.15)
project(NanoEngine VERSION 0.1.0 LANGUAGES CXX)
find_package(OpenMP REQUIRED)

# We use C++20 for modern memory management and concepts
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Optimization flags: -O3 is critical for inference speed
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -march=native")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -march=native -mavx2")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -march=native -mavx2 ${OpenMP_CXX_FLAGS}")
# Include directories
include_directories(include)
include_directories(include third_party)

# Define our main executable
add_executable(nano_engine src/main.cpp src/tensor.cpp src/ops.cpp src/loader.cpp src/model.cpp src/tokenizer.cpp src/safetensors.cpp)

# (Optional) For later: Add subdirectories for tests
# enable_testing()