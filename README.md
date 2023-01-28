# TWO NESTED CASCADE CLASSIFIERS FOR REAL-TIME MULTI-VIEW FACE DETECTION IN UNCONSTRAINED REAL-WORLD CONDITIONS

This program loads a cascade of nested classifiers (XML file) constructed by [UVtrainer](https://github.com/roggerfq/multiview-npd/tree/master/UVtrainer). Then, the algorithm can detect faces with pose changes between [-90º, 90º] CROP (rotation-off-plane).

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

You can change the default video-input device at line 6 in ./two-Nested-NPD_Multi-view/main.cpp file.

## Demo Video

[![Watch the video](https://github.com/roggerfq/multiview-npd/blob/master/two-Nested-NPD_Multi-view/results/face_detection_demo.png)](https://www.youtube.com/watch?v=sSboyjU7WUc)

[![Watch the video](https://github.com/roggerfq/multiview-npd/blob/master/two-Nested-NPD_Multi-view/results/face_detection_demo2.png)](https://www.youtube.com/watch?v=CRdJJsVQ7cc)

## License

This project is licensed under the MIT License

## Acknowledgments
* Program FORMACIÓN E INNOVACIÓN PARA EL FORTALECIMIENTO DE LA COMPETITIVIDAD DEL SECTOR TIC DE LA REGIÓN: FORMATIC E INNOVATIC VALLE DEL CAUCA, OCCIDENTE supported by InfiValle, Gobernación del Valle del Cauca, and PacifiTIC.
* NVIDIA Corporation
* Comisión Fulbright Colombia
