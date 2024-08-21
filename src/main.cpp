#include "util/Config.hpp"
#include <cstdlib>
#include <iostream>
#include "core/mandelbrot.hpp"

int main() {
    Config cfg("../settings.cfg");

    renderLoop();

    return EXIT_SUCCESS;
}
