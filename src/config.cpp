#include "config.hpp"
#include "json.hpp"
#include <fstream>

using json = nlohmann::json;

void Config::load(const std::string& path) {
    std::ifstream f(path);
    if (!f.is_open()) {
        std::err<<"Error: Could not open the config file" << path << std::endl;
    }
}