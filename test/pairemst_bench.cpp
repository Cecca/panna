//! Head-to-head benchmark: `EMST::find_tree` (the `Index`/`PrefixMap` search
//! driven by a persistent thread pool) against `pair_forest_emst` (the
//! `PairForestIndex` search driven by OpenMP), in one process, on the same
//! data, with the same seed and the same parameters.
//!
//!   ./pairemst-bench [epsilon]     # epsilon defaults to 0.2
//!
//! The two entry points take their input differently: `EMST` is handed the raw
//! `std::vector<std::vector<float>>` and builds its own `Index` (and hence its
//! own `Dataset`) from it, while `pair_forest_emst` takes a ready-made
//! `Dataset`. We therefore build the `NormedPoints` for the new path by hand
//! from the very same vector, in the very same order, so that the two searches
//! see identical point ids and the weights are directly comparable.

#include <highfive/H5Easy.hpp>

#include <chrono>
#include <cstdlib>
#include <string>
#include <vector>

#include "panna/data.hpp"
#include "panna/distance.hpp"
#include "panna/emst.hpp"
#include "panna/logging.hpp"
#include "panna/lsh/lattice.hpp"
#include "panna/pairemst.hpp"
#include "panna/rand.hpp"

using namespace panna;

int main( int argc, char** argv ) {
    seed_global_rng( 365 );

    const float epsilon = ( argc > 1 ) ? std::strtof( argv[1], nullptr ) : 0.2f;
    const size_t rep = 512;
    const float delta = 0.01f;

    using Dataset = NormedPoints;
    using Distance = EuclideanDistance;
    using Hasher = LatticeLSH<4, Dataset, Distance>;

    H5Easy::File file( "datasets/glove-100-normalized.hdf5", H5Easy::File::ReadOnly );
    // H5Easy::File file( "datasets/fashion-mnist-784-euclidean.hdf5", H5Easy::File::ReadOnly );
    std::vector<std::vector<float>> points =
        H5Easy::load<std::vector<std::vector<float>>>( file, "/train" );
    points.resize(50000);

    const size_t dimensions = points[0].size();
    // clang-format off
    LOG_INFO( "msg", "pairemst benchmark",
              "n", points.size(),
              "dimensions", dimensions,
              "repetitions", rep,
              "delta", delta,
              "epsilon", epsilon );
    // clang-format on

    // --- baseline: EMST::find_tree ------------------------------------------
    double baseline_seconds = 0.0;
    float baseline_weight = 0.0f;
    size_t baseline_distances = 0;
    size_t baseline_bytes = 0;
    {
        EMST<Dataset, Hasher, Distance> baseline( dimensions, rep, points, delta, epsilon );
        const auto start = std::chrono::steady_clock::now();
        const auto [weight, tree] = baseline.find_tree();
        const auto end = std::chrono::steady_clock::now();

        baseline_seconds = std::chrono::duration<double>( end - start ).count();
        baseline_weight = weight;
        baseline_distances = baseline.get_distance_count();
        baseline_bytes = baseline.get_index_size_bytes();
    }

    // --- the new search: pair_forest_emst -----------------------------------
    /// Re-seed before the second run. `get_global_rng` is one function-local
    /// static, so without this the new search would start from whatever state
    /// the baseline left behind and the two paths would draw different hash
    /// functions and a different `kcenter` seed -- the comparison would not be
    /// over the same configuration.
    seed_global_rng( 365 );

    // Same points, same order, so the two trees are over the same vertex set.
    Dataset data( dimensions );
    for ( const auto& point : points ) {
        data.push_back( point.begin(), point.end() );
    }

    const auto start = std::chrono::steady_clock::now();
    const PairEmstResult result =
        pair_forest_emst<Dataset, Hasher, Distance>( data, epsilon, delta, rep );
    const auto end = std::chrono::steady_clock::now();
    const double new_seconds = std::chrono::duration<double>( end - start ).count();

    // clang-format off
    LOG_INFO( "msg", "baseline EMST::find_tree",
              "elapsed_s", baseline_seconds,
              "weight", baseline_weight,
              "distances_computed", baseline_distances,
              "index_bytes", baseline_bytes );
    LOG_INFO( "msg", "pair_forest_emst",
              "elapsed_s", new_seconds,
              "weight", result.weight,
              "distances_computed", result.distances_computed,
              "prefix_at_stop", result.prefix_at_stop,
              "repetitions_at_stop", result.repetitions_at_stop,
              "index_bytes", result.index_bytes );
    LOG_INFO( "msg", "comparison",
              "weight_ratio", result.weight / baseline_weight,
              "speedup", baseline_seconds / new_seconds );
    // clang-format on

    return 0;
}
