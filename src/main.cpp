#include "util/Config.hpp"
#include <cstdlib>
#include <iostream>
#include "core/mandelbrot.hpp"

int main() {
    Config cfg("/home/hhuebner/Documents/mandelspace/settings.cfg");

    std::cout << cfg.test3  << std::endl;
    render_mandelbrot();
    return EXIT_SUCCESS;
}