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


## Overview

This project is a **CUDA-accelerated [path tracer](https://en.wikipedia.org/wiki/Path_tracing)** which simulates global illumination through physically-based lighting and material techniques, with the goal of generating photorealistic renders. It presents various features, including **physically-based materials**, **custom 3D model loading**, and an efficient **BVH geometry acceleration structure** with surface-area based construction for high-performance traversal. The renderer itself incorporates **stochastic sampled anti-aliasing**, **path partitioning** for active ray optimization, and various other rendering optimizations. These optimizations allow it to render extremely dense scenes like the **Stanford Dragon** (800K triangles) in complex lighting scenarios, all within a reasonable amount of time while maintaining physically accurate light transport. Most scenes converge at about **400 iterations**.

## Code Structure

Code is available in `src/`. Each file generally has a `.h` and `.cu`/`.cpp` pair. The following files are of notable importance:

```py
project-root/
├── shaders/
│   ├── lambert.cu              # Diffuse BRDF, PDF, Sampling function
│   ├── specular.cu             # Reflection BRDF, mirror sampling
│   └── cook_torrance.cu        # Microfacet BRDF, GGX sampling
├── common.cu               # Useful CUDA-specific macros
├── intersections.cu        # Intersection logic for primitives, BVH traversal
├── mesh.cu                 # Mesh struct, BVH construction
├── pathtrace.cu            # The main pathtracer loop and kernels
├── stack.cu                # A device-side compile-time register stack
├── scene.cpp               # Handles parsing .json scenes and mesh loading
├── sceneStructs.h          # Structs for pathtracer elements e.g. Ray, PathSegment 
└── main.cpp
```

`scenes/` contains a few scenes specified in a JSON format. Some of them reference obj models, which exist in `obj/`.

