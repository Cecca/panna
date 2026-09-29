# PANNA: Playground for Approximate Nearest Neighbor Algorithms

This library aims at providing useful building blocks to implement algorithms for approximate nearest neighbor search.

## Building

This is, first and foremost, a header only library requiring `C++17` and depending on [`cereal`](https://uscilab.github.io/cereal/index.html) and [`ffht`](https://github.com/FALCONN-LIB/FFHT) (both libraries are vendored in `external`).
To integrate with other codebases simply place `include/panna` in your include path, while making sure that the headers of the dependencies (i.e. the contents of `external`) are included as well.

That said, the repository includes tests and [examples](https://github.com/Cecca/panna/tree/main/examples), which are built using `cmake` with the usual steps

```
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make
```

This produces the following executables:

- `build/test` to run the tests
- `build/glove` to run the example on [`glove`](http://ann-benchmarks.com/glove-100-angular.hdf5)
- `build/fashion` to run the example on [`fashion-mnist`](http://ann-benchmarks.com/fashion-mnist-784-euclidean.hdf5)
