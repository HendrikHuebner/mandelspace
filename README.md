# MandelSpace - 3D fractal renderer using CUDA

Renders 3D fractals in real time. Lighting and different settings can be configured in settings.cfg

Installation:

1. Install GLEW (apt install libglew2.2)
2. Install GLFW (apt install libglfw3)
3. configure cmake (cmake . -B ./build)
4. build (cmake --build ./build)
   
Usage:
The first argument of the program is the config i.e.: ./mandelspace myconfig.cfg

- Move the camera using W, A, S, D, space for up, shift for down
- Rotate the camera using the arrow keys
- Press Enter to reload settings from the config file

## TODO:

- [x] Mandelbulb
- [x] Menger Sponge / prison
- [ ] Sierpinski Pyramid
- [x] Mandelbox

- [x] Raymarching
- [x] Camera movement
- [x] Surface normals
- [x] Basic lighting (diffuse, specular, ambient)
- [x] Cast shadows
- [x] Ambient occlusion
- [x] Anti Aliasing

