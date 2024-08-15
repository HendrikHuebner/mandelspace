#include "util/Config.hpp"
#include <cstdlib>
#include <iostream>

int main() {
    Config cfg("/home/hhuebner/Documents/mandelspace/settings.cfg");

    std::cout << cfg.test3  << std::endl;

    return EXIT_SUCCESS;
}