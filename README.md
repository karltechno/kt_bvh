# kt_bvh

Experimenting with BVH building algorithms and node formats for CPU/GPU raytracing.

The library is not restricted to any particular traversal or node format. It builds an unoptimized pointer based tree which can then be flattened to a more optimal structure. This way it is easier to experiment with different node formats (memory layout, quantization, packing etc). However there are sample routines for building a simple flat BVH2 and BVH4. 

Currently implemented:
* BVH2/4
* Median split
* Binned surface area heuristic split

TODO:
* [SBVH](https://www.nvidia.com/docs/IO/77714/sbvh.pdf) (spatial splits)
* Sweep SAH
* Optimize tree building (multithreading/simd)
* LBVH
