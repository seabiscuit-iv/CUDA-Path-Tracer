CUDA Path Tracer
================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 3**

* Saahil Gupta
  * [LinkedIn](https://www.linkedin.com/in/saahil-g), [personal website](https://www.saahil-gupta.com)
* Tested on: Windows 11 10.0.26100, AMD Ryzen 9 7940HS @ 4.0GHz 32GB, RTX 4060 Laptop GPU 8GB

## Table of Contents

- [Overview](#overview)
  - [Code Structure](#code-structure)
- [Features](#features)
  - [Materials](#materials)
    - [Lambertian Diffuse](#lambertian-diffuse)
    - [Perfectly Specular](#perfect-specular-reflection)
    - [Cook-Torrance PBR](#cook-torrance-pbr-material)
  - [Scene and Geometry Handling](#scene-and-geometry-handling)
    - [Triangle Mesh Rendering](#triangle-mesh-rendering)
    - [OBJ Model Importing](#obj-model-importing)
    - [Custom Normal Buffers](#custom-normal-buffers)
  - [Acceleration Structures](#acceleration-structures)
    - [Bounding Volume Hierarchy](#bounding-volume-hierarchy)
      - [Construction](#construction)
      - [Traversal](#traversal)
    - [Surface Area Heuristic](#surface-area-heuristic)
      - [Binning](#binning)
      - [Traversal Optimizations](#traversal-optimizations)
    - [Stack Height Optimization](#stack-height-optimization)
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

[Perfectly specular reflection](https://pbr-book.org/3ed-2018/Reflection_Models/Specular_Reflection_and_Transmission) is essentially a **mirror**, where any light that comes in is reflected perfectly across the surface normal and not distributed at all. As a result, this material has no editable properties outside of albedo.

The PDF for a mirror would be infinity for the reflected direction of our view vector across the surface normal, and zero everywhere else. This is called a [Dirac delta distribution](https://en.wikipedia.org/wiki/Dirac_delta_function). Similarly, the BRDF ends up also being a Dirac delta distribution. We only trace the reflected ray (since every other ray has no contribution), and so in our code, we omit the infinite values and simply multiply by our surface color. Note that we don't need to multiply by the Lambertian term because the BRDF for perfect specular already includes the cosine implicitly in its definition of a Dirac delta.

The code for this material's BRDF, PDF, sampling and shading logic can be found in `shaders/specular.cu`. 

### Cook-Torrance PBR Material

Lambertian and perfectly specular materials represent the two ends of the diffuse-specular spectrum of materials. However, most surfaces tend to fall somewhere in the middle, having a somewhat distributed yet still concentrated reflection lobe. This is what creates the rough and blurry reflections that are visible on polished and metallic surfaces.

To represent this on metallic materials, we use the [Cook-Torrance microfacet model](https://graphicscompendium.com/gamedev/15-pbr), which consists of a number of infinitely small perfectly specular mirrors, all angled a random distance away from the surface macro-normal. Computing the BRDF and PDF for this model requires the use of various other credited formulas:

- Trowbridge-Reitz Normal Distribution Function (NDF)
- [Schlick Fresnel Approximation](https://en.wikipedia.org/wiki/Schlick%27s_approximation)
- Smith GGX Microfacet Geometry Model

For non-metallic materials, we use the [Dieletric model](https://www.pbr-book.org/4ed/Reflection_Models/Dielectric_BSDF), which similarly uses a roughness value to shift between a clearcoat surface and a more standard diffuse surface. 

Finally to combine these two, we use the material's metallic value to adjust the Fresnel term between the standard Dieletric value of ~0.04, and the Schlick approximation we use for metallics. In our sampling function, we also use the metallic value to adjust a probability value `probSpecular`, which decides the probability of our ray importance sampling either the dieletric or microfacet surface. 

Combining these two gives us a standard PBR material that can be controlled by its albedo, roughness, and metallic values. 

The code for this material's BRDF, PDF, sampling and shading logic can be found in `shaders/cook_torrance.cu`. 

## Scene and Geometry Handling

### Triangle Mesh Rendering

In order to support custom model loading, simple triangle mesh rendering was added early in the project, following the [Möller–Trumbore intersection algorithm](https://en.wikipedia.org/wiki/M%C3%B6ller%E2%80%93Trumbore_intersection_algorithm). Originally this would naively loop through all triangles, but this system would eventually be replaced by a high-performant bounding volume hierarchy.

### OBJ Model Importing

For more customizable scenes, 3D models made in other programs can be imported and rendered through the `.obj` file format. To handle this, this project uses [tinyobjloader](https://github.com/tinyobjloader/tinyobjloader), the files for which can be found in `src/tinyobj`. The program doesn't any pre-rendering input assembly in order to optimize GPU memory usage: both the vertex position data and triangle index buffers are pushed to the GPU exactly as they are read in.

### Custom Normal Buffers

The renderer also supports passing in custom vertex normals, which have no input assembly and are also accessed via a separate set of normal indices. The normals are interpolated with barycentric coordinates, allowing us to use models with smooth shading.


## Acceleration Structures

For triangle meshes and imported models, naive intersection testing consists of iterating and testing every triangle. While this is fine for small models, larger models can have upwards of 500K triangles, and this quickly becomes unsatisfiable. Efficient ray tracing relies on spatial acceleration structures to quickly eliminate large portions of the scene from intersection testing. This project implements a [Bounding Volume Hierarchy (BVH)](https://en.wikipedia.org/wiki/Bounding_volume_hierarchy) to achieve logarithmic traversal performance relative to the number of triangles. 

### Bounding Volume Hierarchy

The BVH organizes the geometry into a tree, where each leaf node represents a single triangle of our original mesh, and every other node is a axis-aligned bounding box (AABB) that encapsulates all of its children nodes. 

#### Construction

The BVH is constructed recursively by partitioning the primitives and sending them over to the left/right children nodes. The tree construction is rather [standard](https://en.wikipedia.org/wiki/Binary_tree), and room for optimization can mainly be found in the partitioning and storage approaches. Originally, we simply partitioned nodes down the center of the range, sending a near-equal amount to the left and right children. This resulted in a complete binary tree, which had optimal tree height. Another benefit to this method was the ability to store the tree is a heap-style array, where a node at index $i$ had children at $2i+1$ and $2i+2$. However, due to the nature of the BVH traversal algorithm, this resulted in a very unoptimal BVH as a whole. This method was eventually replaced with an [SAH](#surface-area-heuristic) based partition.

#### Traversal

During ray traversal, the BVH is explored top-down in a [depth-first search](https://en.wikipedia.org/wiki/Depth-first_search) manner. The goal is to track and find the primitive that is closest to the ray's origin. Ray-AABB intersections are tested first, and only the branches that intersect are recursively visited. Leaf nodes then perform ray-triangle intersection tests. A number of low-hanging fruits can be picked here to promote early branch pruning, detailed in [traversal optimizations](#traversal-optimizations). 


### Surface Area Heuristic

The Surface Area Heuristic, or SAH, is a formula that estimates the **cost** of splitting a set of primitives into two child nodes by considering the surface areas of the resulting AABBs and their primitive counts.

$$
  C = C_{trav} + \frac{A_L}{A_P}N_L C_{L} + \frac{A_R}{A_P}N_R C_{R}
$$
- $C$ - total cost of this split
- $A_L, A_R$ - Surface area of the left and right partitions bounding box
- $N_L, N_R$ - Number of primitives in the left and right partition
- $C_L, C_R$ - The cost of traversing left and right children

This equation balances traversal speed against build cost, resulting in efficient hierarchies for complex meshes. Our goal when constructing our BVH is to **minimize** $C$, which would result in an optimal split. 

To compute the SAH, we could loop through every possible split, of which there are $n$ for a single axis. However, since computing the SAH takes $O(n)$, the total BVH construction rumtime would be $O(n^3\log n)$, which is unsatisfiable for large meshes. Therefore, we use a construction optimization called [binning](#binning).

Furthermore, since we construct the tree top-down, it's impractical to compute $C_L$ and $C_R$. Therefore, we omit them from our SAH calculation (including $C_{trav}$ for consistency), and instead use a different [stack height optimization](#stack-height-optimization) to avoid blowing up our tree height.


#### Binning

Binning is a way to improve the speed of minimizing the SAH at every level. The naive way to minimize the SAH is to calculate it at the centroid of every primitive, since this represents every possible split that can be made along an axis. Instead of doing this, we instead compute the SAH at constant intervals, allowing us to "bin" the primitives into interavls and precompute the bounding box of each bin. This reduces our SAH calculating runtime to $O(n)$, and our total tree construction time to $O(n^2\log n)$.


#### Traversal Optimizations

In our traversal algorithm, there are a number of quick and dirty optimizations we can make that will promote early branch pruning and have a strong impact on performance

- **Min-Distance Termination**  
If we intersect a bounding box, only traverse down if the intersection distance is closer than the current tracked primitive intersection.
- **Node Sorting**  
Sort the child nodes based on their bounding box intersection distances, promoting more min-distance termination


### Stack Height Optimization

Earlier, we mentioned that in our SAH calculation, we do not calculate $C_L$ and $C_R$ due to its impracticality and just assume them to be 1. However, this has consequences, as these cost values were the only defense against creating severely unbalanced partitions, where one child tree has a much larger height than the other.

In order to defend against this, we can use a default case, where if the split point is in the top or bottom 15% of the range, we default to using a midpoint split. This is a quick and dirty solution to this problem, but it provides a good enough result for this project's needs. We can compute a midpoint split on our axis by performing a partition about the $\frac{n}{2}$th statistic. 

This will guarantee that our tree height is no more than $\log _\frac{1}{1 - 0.15} n$, or $\log _{1.176} n$. For 2.5M triangles, this means our tree height is guarateed to be no more than 100. This is extremely important, because it allows us to create a constant-sized stack class (`src/stack.cu`) that will be stored in VPGRs rather than some paged memory module, creating noticeable performance gains.


## Rendering Pipeline Improvements

This project implements several optimizations to improve GPU path tracing throughput and memory coherence. Two key techniques help minimize warp divergence and balance computational load across threads.

### Terminated Path Partitioning

In a naive path tracer, all path segments are processed each iteration, even those that have already terminated (e.g., after hitting a light or not intersecting anything). This wastes GPU lanes and reduces overall efficiency.

**Terminated path partitioning** uses a parallel compaction step via Thrust to partition all active rays to appear before our dead rays in our path data structure. This allows us to reduce the number of blocks we push to the work queue each bounce iteration. This ensures that subsequent kernel launches only operate on active rays, improving occupancy and reducing divergence.


### Material Sorting

When shading, different materials (Lambertian, Specular, Cook-Torrance, etc.) often follow distinct code paths, which can cause warp divergence in CUDA. **Material Sorting** groups path segments by their material type before shading, so threads within a warp execute similar instructions. 