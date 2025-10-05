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
    - Lambertian Diffuse
    - Perfectly Specular
    - Cook-Torrance PBR
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

<br/>

# Overview

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

<br/>

# Features

This path tracer contains a number of advanced features to improve **visual fidelity**, **frame time**, and **convergence time**.

## Materials

### Lambertian Diffuse

A very simple diffuse material BRDF based on [lambertian reflectance](https://en.wikipedia.org/wiki/Lambertian_reflectance). The general idea behind the Lambert shading model is that illumination is inversely related to the cosine of the angle between the surface normal and the incoming light ray, and is independent of the azimuthal angle of the light ray and view ray. Some examples of materials represented well by Lambertian reflectance include **paper, concrete, wood, and paint**.

The Lambert BRDF is only dependent on albedo, so the sampling method we use is simply cosine-weighted hemisphere sampling about the surface normal (to balance out [Lambert's law](https://en.wikipedia.org/wiki/Beer%E2%80%93Lambert_law)). The PDF is just $\frac{cos(\theta)}{\pi}$.

The code for this material's BRDF, PDF, sampling and shading logic can be found in `shaders/lambert.cu`. 

### Perfect Specular Reflection

[Perfectly specular reflection](https://pbr-book.org/3ed-2018/Reflection_Models/Specular_Reflection_and_Transmission) is essentially a **mirror**, where any light that comes in is reflected perfectly across the surface normal and not distributed at all. As a result, this material has no editable properties. Albedo, roughness, and metallic values have *zero* contribution to the final output. 

The PDF for a mirror would be infinity for the reflected direction of our view vector across the surface normal, and zero everywhere else. This is called a [Dirac delta distribution](https://en.wikipedia.org/wiki/Dirac_delta_function). Similarly, the BRDF ends up also being a Dirac delta distribution. We only trace the reflected ray (since every other ray has no contribution), and so in our code, we omit the infinite values and simply multiply by our surface color. Note that we don't need to multiply by the Lambertian term because the BRDF for perfect specular already includes the cosine implicitly in its definition of a Dirac delta.

The code for this material's BRDF, PDF, sampling and shading logic can be found in `shaders/specular.cu`. 