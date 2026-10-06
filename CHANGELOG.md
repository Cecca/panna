# Changelog

All notable changes to this project will be documented in this file.

## [0.1.0] - 2026-10-06

First public release.

### 🚀 Features

#### LSH families (`include/panna/lsh`)

- SimHash for angular distance
- Euclidean (E2) LSH, with data-driven parameter fitting and refitting
- Cross-polytope LSH, using random rotations and the fast Hadamard transform
- Lattice LSH (E8), with collision probability estimation and `fit`
- MinHash for Jaccard distance
- Tensoring of hash functions, with 16-bit hash values
- Failure-probability predicates to decide when to stop a search
- LSH values can be used as keys in hash maps

#### Data and distances

- Data formats for dense vectors, including `UnitNormPoint`/`NormedPoints`
- Euclidean, squared Euclidean, angular and Jaccard distances
- Euclidean distance optimized with AVX2/FMA and loop unrolling
- A GEMM kernel for blocked distance computations
- `RandomDotProducts` helper and convenient random number generation

#### Indices and search

- `TrieIndex` for nearest neighbor search, based on the PUFFINN prefix map
- `PairForestIndex` for finding close pairs of points

#### Euclidean minimum spanning tree and HDBSCAN

- Approximate EMST computed with LSH (`pair_forest_emst`)
- Approximate MST under the mutual reachability distance
  (`pair_forest_emst_mutual_reachability`), which is the basis of HDBSCAN.
  Core distances are sharpened with NN-descent before the LSH search.
- Helpers: `approximate_diameter` and `distance_histogram`
- Statistics on each run, including a timing breakdown

#### Python package (`pypanna`)

- Bindings for the index and EMST functionality, built with nanobind
- `pypanna.knn`: exact k-NN graph computation
- `pypanna.mst`: exact (mutual reachability) MST computation
- `pypanna.datasets`: download and loading of benchmark datasets (ANN-benchmarks
  datasets, PAMAP2, Census, HT, Yandex T2I, Chem, DENSIRED), with removal of
  duplicates and of zero rows, and optional normalization and standardization
- `set_seed` to control randomness, and `git_version` to report the build

#### Utilities

- Structured logging
- `Timer` utility
- Build commit hash embedded in the binaries

### ⚡ Performance

- Parallel search with OpenMP, with reworked parallelism and lower memory use
- Squared distances used wherever the square root is not needed
- Dot products reused while tuning the radius, and edges processed in batches
  to limit memory
- Faster `approximate_diameter` and E8 decoding
- Sampling of points to tune LSH parameters

### 📚 Documentation

- README with installation and build instructions
- Examples in C++ and Python (GloVe, Fashion-MNIST, EMST)
- MIT license

