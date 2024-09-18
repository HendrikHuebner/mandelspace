#include "util/Config.hpp"
#include <cstdlib>
#include <iostream>
#include "window.hpp"

int main(int argc, char* argv[]) {
    std::string settings = "./settings.cfg";
    if (argc > 1) {
        settings = argv[1]; // First argument is the settings path
    }
    Config *cfg = new Config(ConfigParser(settings));

    renderLoop(cfg, settings);

    return EXIT_SUCCESS;
}
