#include "panna/emst.hpp"

#include <highfive/H5Easy.hpp>

#include <chrono>
#include <iostream>
#include <string>

#include "panna/data.hpp"
#include "panna/distance.hpp"
#include "panna/logging.hpp"
#include "panna/lsh/lattice.hpp"
#include "panna/rand.hpp"

using namespace panna;

int main( int argc, char** argv ) {
    if ( argc < 2 ) {
        std::cerr << "usage: " << argv[0] << " <dataset.hdf5> [epsilon]" << std::endl;
        return 1;
   }
    const std::string path( argv[1] );
    const float epsilon = ( argc > 2 ) ? std::stof( argv[2] ) : 0.0f;

    seed_global_rng( 365 );

    // Parameters for the LSH index used by the EMST construction
    const size_t rep = 1024;
    using Dataset = EuclideanPoints;
    using Distance = EuclideanDistance;
    using Hasher = LatticeLSH<4, Dataset, Distance>;

    H5Easy::File file( path, H5Easy::File::ReadOnly );
    std::vector<std::vector<float>> points =
        H5Easy::load<std::vector<std::vector<float>>>( file, "/train" );
    if (points.size() > 100000) {
        points.resize(100000);
    }

    std::cout << "loaded " << points.size() << " points from " << path << std::endl;

    const size_t dimensions = points[0].size();
    const size_t n = points.size();
    const auto start = std::chrono::steady_clock::now();
    const float delta = 0.1;
    EMST<Dataset, Hasher, Distance> tree( dimensions, rep, points, delta, epsilon );

    const auto random = tree.random_emst();
    float random_weight = 0.0;
    for (const auto &e: random) {
        random_weight += e.weight;
    }
    const auto [rand_prefix, rand_rep] = *tree.table.earliest_confirming_iteration(random.back().weight, delta/n);
    LOG_INFO(
      "msg", "tree info",
      "maximum-random-weight", random.back().weight,
      "total-random-weight", random_weight,
      "earliest-confirm-prefix", rand_prefix,
      "earliest-confirm-repetition", rand_rep
    );

    const auto& [weight, emst] = tree.find_tree();
    const auto end = std::chrono::steady_clock::now();
    const double elapsed_s = std::chrono::duration<double>( end - start ).count();

    const auto [opt_prefix_min, opt_rep_min] = *tree.table.earliest_confirming_iteration(emst.front().weight, delta/n);
    const auto [opt_prefix, opt_rep] = *tree.table.earliest_confirming_iteration(emst.back().weight, delta/n);
    LOG_INFO(
      "msg", "tree info",
      "maximum-opt-weight", emst.back().weight,
      "total-opt-weight", weight,
      "earliest-confirm-prefix-min", opt_prefix_min,
      "earliest-confirm-repetition-min", opt_rep_min,
      "earliest-confirm-prefix", opt_prefix,
      "earliest-confirm-repetition", opt_rep,
      "running-time", elapsed_s
    );

    return 0;
}
