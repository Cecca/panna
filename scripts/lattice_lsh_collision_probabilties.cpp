#include <panna/data.hpp>
#include <panna/lsh/lattice.hpp>
#include <panna/git_version.hpp>
#include <format>
#include <iostream>
#include <fstream>
#include <ctime>
#include <random>

/// Estimate the collision probabilities of
/// the LatticeLSH family with the E8 lattice.
/// Writes the output to a header file that can be included
/// by LatticeLSH itself to easily report collision probabilities
int main( int, char** ) {
    using namespace panna;
    using HashFamily = LatticeLSH<1, EuclideanPoints, EuclideanDistance>;

    const size_t samples = 1e6;
    const size_t dimensions = 8;
    const float step = 0.01;
    const float max_distance = 6.0;

    const size_t num_distances = static_cast<size_t>(std::ceil(max_distance / step));
    std::vector<float> probabilities(num_distances);
    probabilities[0] = 1.0;

    #pragma omp parallel for
    for ( size_t i = 1; i < num_distances; i++ ) {
        std::mt19937_64 rng( i );

        std::vector<HashFamily::Value> h1;
        std::vector<HashFamily::Value> h2;

        float distance = i * step;
        EuclideanPoints pts( dimensions );
        auto x = sample_random_normal_vector( dimensions, rng );
        auto direction = sample_random_normal_vector( dimensions, rng );
        normalize( direction );
        rescale( direction, distance );
        auto y = add( x, direction );

        pts.push_back( x.begin(), x.end() );
        pts.push_back( y.begin(), y.end() );
        const float d = EuclideanDistance::compute( pts[0], pts[1] );
        expect( std::abs( distance - d ) < 1e-5 );

        std::vector<float> zero(dimensions);
        HashFamily lsh( zero, 1, dimensions, samples, rng );
        lsh.hash( pts[0], h1 );
        lsh.hash( pts[1], h2 );
        size_t collisions = 0;
        for ( size_t rep = 0; rep < samples; rep++ ) {
            if ( h1[rep] == h2[rep] ) {
                collisions++;
            }
        }
        float empirical_p = ( (float)collisions ) / samples;
        LOG_INFO( "distance", distance, "p", empirical_p );
        probabilities[i] = empirical_p;
    }

    std::time_t timestamp;
    time( &timestamp );
    std::ofstream output( "include/panna/lsh/lattice_probabilities.hpp" );

    output << std::format(R"(#pragma once
// created with git version {}
// on {}

#include <cstddef>

namespace panna::lattice_lsh {{
    static const float DISTANCE_STEP = {};
    static const float MAX_DISTANCE = {};
    static const size_t SAMPLES = {};

    static const float PROBABILITIES[] = {{)", GIT_COMMIT_HASH, std::ctime(&timestamp), step, max_distance, samples) << std::endl;

    for (float p : probabilities) {
        output << std::format("        {:.9f},", p) << std::endl;
    }

    output << "    };" << std::endl
           << "    static const size_t NUM_ESTIMATES = " << probabilities.size() << ";" << std::endl
           << "}" << std::endl;
    output.close();
}
