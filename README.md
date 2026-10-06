# PANNA: Playground for Approximate Nearest Neighbor Algorithms

This repository aims at providing useful building blocks to implement algorithms for approximate nearest neighbor search.

The repository provides mainly two things:

- A software library to build similarity search algorithms, implemented mainly as a C++ library, with python bindings
- An environment to run experiments, pinning versions and dependencies of related baselines by means of Nix flakes

## Python: installing

The library should be easily installable with the usual

```
pip install pypanna
```

If you are using `uv` to manage your dependencies

```
uv add pypanna
```

should suffice.

## C++: Building

This is, first and foremost, a header only library requiring `C++20` and depending on [`ffht`](https://github.com/FALCONN-LIB/FFHT) (vendored in `external`).
To integrate with other codebases simply place `include/panna` in your include path, while making sure that the headers of the dependency (i.e. the contents of `external`) are included as well.

That said, the repository includes tests and [examples](https://github.com/Cecca/panna/tree/main/examples), which are built using `cmake` with the usual steps

```
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make
```
