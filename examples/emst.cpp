#include "panna/lsh/crosspolytope.hpp"
#include "panna/lsh/simhash.hpp"
#define EXPECT_ACTIVE
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
    const size_t rep = 512;
    using Dataset = UnitNormPoints;
    using Distance = CosineDistance;
    using Hasher = Simhash<24, Dataset, Distance>;
    // using Dataset = EuclideanPoints;
    // using Distance = EuclideanDistance;
    // using Hasher = LatticeLSH<4, Dataset, Distance>;

    H5Easy::File file( path, H5Easy::File::ReadOnly );
    std::vector<std::vector<float>> points =
        H5Easy::load<std::vector<std::vector<float>>>( file, "/train" );
    if (points.size() > 100000) {
        points.resize(100000);
    }

    std::cout << "loaded " << points.size() << " points from " << path << std::endl;

    const size_t dimensions = points[0].size();
    const auto start = std::chrono::steady_clock::now();
    const float delta = 0.1;
    EMST<Dataset, Hasher, Distance> tree( dimensions, rep, points, delta, epsilon );

    const auto& [emst, cored] = tree.find_tree_mutual_reachability_distance(5);
    const auto end = std::chrono::steady_clock::now();
    const double elapsed_s = std::chrono::duration<double>( end - start ).count();

    LOG_INFO(
      "msg", "tree info",
      // "maximum-opt-weight", emst.back().weight,
      // "total-opt-weight", weight,
      "running-time", elapsed_s
    );

    return 0;
}
