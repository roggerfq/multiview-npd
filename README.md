# Two Nested Cascade Classifiers for Real-Time Multi-View Face Detection in Unconstrained Environments

This repository implements a **real-time multi-view face detection system** based on the work presented in [1].

To train the detector, we designed a dedicated training framework called **[UVtrainer](https://github.com/roggerfq/UVtrainer)**, which enables the training of generic object detectors in a flexible and efficient manner.

After training, **UVtrainer** generates an **XML model file**, which is then loaded by the detection algorithm implemented in this repository.

Unlike the original work, this implementation was specifically designed to achieve robust face detection under large in-plane rotations in the range of **[-90°, 90°]**.

Quantitative performance metrics and additional experimental results can be found in the **Results** section of **[UVface](https://github.com/roggerfq/UVface)**.

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

## Author
Roger Figueroa Quintero - [LinkedIn Profile](https://www.linkedin.com/in/roger-figueroa-quintero/)

---

## License
This project is licensed under the [MIT License](LICENSE.md), allowing unrestricted use, modification, and distribution under the terms of the license.


---

## Acknowledgments

This work was supported by:

- **FORMACIÓN E INNOVACIÓN PARA EL FORTALECIMIENTO DE LA COMPETITIVIDAD DEL SECTOR TIC DE LA REGIÓN**  
  (FORMATIC e INNOVATIC Valle del Cauca, Occidente), supported by **InfiValle**, **Gobernación del Valle del Cauca**, and **PacifiTIC**
- **NVIDIA Corporation**
- **Comisión Fulbright Colombia**

## References

[1] S. Liao, A. K. Jain, and S. Z. Li, "A fast and accurate unconstrained face detector," *IEEE Trans. Pattern Anal. Mach. Intell.*, vol. 38, no. 2, pp. 211-223, 2015.
