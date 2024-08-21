#include "util/Config.hpp"
#include <cstdlib>
#include <iostream>
#include "window.hpp"
#include "fractal.hpp"

int main() {
    Config cfg("../settings.cfg");

    Mandelbulb frac(8, 8);
    renderLoop(frac);

    return EXIT_SUCCESS;
}
