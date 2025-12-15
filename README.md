# Two Nested Cascade Classifiers for Real-Time Multi-View Face Detection in Unconstrained Environments

This project implements a real-time multi-view face detection system based on a cascade of **nested classifiers** stored in an XML file. The classifier is trained using [UVtrainer](https://github.com/roggerfq/UVtrainer).

The system is designed to robustly detect faces under significant pose variations, covering out-of-plane rotations in the range of **[-90°, 90°]**, making it suitable for unconstrained real-world scenarios.

---

## Prerequisites

- OpenCV **>= 2.4**
- CMake **>= 2.8**
- A C++ compiler compatible with your OpenCV version

---

## Compilation

From the project root directory:

```bash
cd two-Nested-NPD_Multi-view
mkdir build
cd build
cmake ..
make
```

---

## Getting Started

After successful compilation, run the application from the `build` directory:

```bash
./main
```

To change the default video input device, modify **line 6** in the following file:

```
two-Nested-NPD_Multi-view/main.cpp
```

---

## Demo Videos

Click on the images below to watch the demo videos:

[![Face Detection Demo 1](https://github.com/roggerfq/multiview-npd/blob/master/two-Nested-NPD_Multi-view/results/face_detection_demo.png)](https://www.youtube.com/watch?v=sSboyjU7WUc)

[![Face Detection Demo 2](https://github.com/roggerfq/multiview-npd/blob/master/two-Nested-NPD_Multi-view/results/face_detection_demo2.png)](https://www.youtube.com/watch?v=CRdJJsVQ7cc)

---

## License

This project is licensed under the **MIT License**.

---

## Acknowledgments

This work was supported by:

- **FORMACIÓN E INNOVACIÓN PARA EL FORTALECIMIENTO DE LA COMPETITIVIDAD DEL SECTOR TIC DE LA REGIÓN**  
  (FORMATIC e INNOVATIC Valle del Cauca, Occidente), supported by **InfiValle**, **Gobernación del Valle del Cauca**, and **PacifiTIC**
- **NVIDIA Corporation**
- **Comisión Fulbright Colombia**
