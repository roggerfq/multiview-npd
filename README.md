# TWO NESTED CASCADE CLASSIFIERS FOR REAL-TIME MULTI-VIEW FACE DETECTION IN UNCONSTRAINED REAL-WORLD CONDITIONS

This program load a nested classification cascade constructed by [UVtrainer](https://github.com/roggerfq/UVtrainer/tree/master/UVtrainer) and performs face detection on faces with pose changes between [-90º, 90º] ROP (rotation-off-plane).

## Prerequisites

OpenCV >= 2.4

CMake version >= 2.8

### Compilation

cd two-Nested-NPD_Multi-view

mkdir build

cd build

cmake ..

make

### Getting Started

In build directory run: ./main 

You can change the default video-input device at line 6 in main.cpp file.

## Demo Video

[![Watch the video](https://github.com/roggerfq/UVtrainer/blob/master/two-Nested-NPD_Multi-view/results/face_detection_demo.png)](https://drive.google.com/file/d/1rTnx-kSE7PPMJmGL6viTbBOAIFw_fCLg/view?usp=sharing)

## License

This project is licensed under the MIT License

## Acknowledgments
* Program FORMACIÓN E INNOVACIÓN PARA EL FORTALECIMIENTO DE LA COMPETITIVIDAD DEL SECTOR TIC DE LA REGIÓN: FORMATIC E INNOVATIC VALLE DEL CAUCA, OCCIDENTE supported by InfiValle, Gobernación del Valle del Cauca, and PacifiTIC.
* NVIDIA Corporation
* Comisión Fulbright Colombia
