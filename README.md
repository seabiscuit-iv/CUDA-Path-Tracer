CUDA Path Tracer
================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 3**

* Saahil Gupta
  * [LinkedIn](https://www.linkedin.com/in/saahil-g), [personal website](https://www.saahil-gupta.com)
* Tested on: Windows 11 10.0.26100, AMD Ryzen 9 7940HS @ 4.0GHz 32GB, RTX 4060 Laptop GPU 8GB

## Table of Contents

- Overview
  - Code Structure
- Features
  - Materials
    - Lambertian Diffuse Materials
    - Perfectly Specular Materials
    - Cook-Torrance PBR Materials
  - Scene and Geometry Handling
    - Triangle Mesh Rendering
    - OBJ Model Importing
    - Custom Normal Buffers
  - Acceleration Structures
    - Bounding Volume Hierarchy
      - Construction
      - Traversal
    - Surface Area Heuristic
    - Traversal Optimizations
      - Min-distance Termination
      - Node Sorting
    - Stack Height Optimization
  - Rendering Pipeline Improvements
    - Terminated Path Partitioning
    - Material Sorting
- Performance Analysis
  - Bounding Volume Hierarchy
  - Terminated Path Partitioning
  - Material Sorting
- Renders
- Future Goals
- Miscallaneous Lessons
  - Firefly Reduction
  - FPU Operation Intrinsics
  - Caching Division Values
- References