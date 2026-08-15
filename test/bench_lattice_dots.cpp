// Microbenchmark: LatticeLSH::hash with and without `RandomDotProducts`.
//
// `LatticeLSH` needs `repetitions * K * 8` random projections of every point it
// hashes. Rather than storing a projection matrix it uses `RandomDotProducts`,
// which produces all of them at once with three rounds of (random +-1 diagonal,
// fast Hadamard transform) over a buffer padded to the next power of two of
// `max(dimensions, repetitions * K * 8)`.
//
// This benchmark pits that against the textbook alternative: an explicit dense
// matrix of Gaussian directions, one dot product per projection. Everything
// downstream of the projections (the bias FMA and the E8 decoding) is identical
// in the two variants, so the difference in the reported timings is exactly what
// `RandomDotProducts` buys - or costs.
//
//   ./bench_lattice_dots [dimensions] [repetitions]

#include "panna/data.hpp"
#define ANKERL_NANOBENCH_IMPLEMENT
#include <nanobench.h>
#include "panna/distance.hpp"
#include "panna/linalg.hpp"
#include "panna/lsh/lattice.hpp"
#include "panna/lsh/values.hpp"
#include "panna/rand.hpp"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

namespace {

    using panna::EuclideanDistance;
    using panna::EuclideanPointHandle;
    using panna::EuclideanPoints;

    //! Fixed by the task: 4 concatenated lattice codes per repetition.
    constexpr uint8_t K = 4;
    //! The E8 lattice lives in 8 dimensions, as in `LatticeLSH`.
    constexpr size_t LATTICE_DIMENSIONS = 8;
    //! Seed so that repeated runs measure the very same random instance.
    constexpr uint64_t SEED = 1234567;
    //! How many distinct points we cycle through, so that nothing gets folded
    //! away and the input is not permanently sitting in L1.
    constexpr size_t NUM_POINTS = 64;

    using ReferenceHasher = panna::LatticeLSH<K, EuclideanPoints, EuclideanDistance>;
    using Value = panna::LongLshValue<K>;
    static_assert( std::is_same<ReferenceHasher::Value, Value>::value,
                   "the two hashers must produce the same value type" );

    //! The same hash function as `LatticeLSH`, but computing the projections
    //! against an explicit matrix of Gaussian directions instead of going
    //! through `RandomDotProducts`.
    class DenseLatticeLSH {
        //! `num_projections` rows of `dimensions` floats each, row major.
        std::vector<float> matrix;
        //! One random offset per projection, like `LatticeLSH::projection_bias`.
        std::vector<float> projection_bias;
        //! Scratch space holding the projections of the point being hashed.
        std::vector<float> projections;
        float scaling_factor;
        size_t dimensions;
        size_t repetitions;

    public:
        using Value = ::Value;

        DenseLatticeLSH( float scaling_factor, size_t dimensions, size_t repetitions ):
            scaling_factor( scaling_factor ),
            dimensions( dimensions ),
            repetitions( repetitions ) {

            const size_t num_projections = repetitions * K * LATTICE_DIMENSIONS;

            // `LatticeLSH` scales the output of `RandomDotProducts` by
            // 1/sqrt(LATTICE_DIMENSIONS); folding the same factor into the
            // entries of the matrix puts the projections of the two variants on
            // the same scale, so the data dependent cost of `decode_e8_code`
            // (which picks the closer of two cosets) is comparable.
            const float entry_scale = 1.0f / std::sqrt( static_cast<float>( LATTICE_DIMENSIONS ) );
            matrix.reserve( num_projections * dimensions );
            for ( size_t i = 0; i < num_projections * dimensions; i++ ) {
                matrix.push_back( panna::sample_random_normal() * entry_scale );
            }

            // The `corrections` term of `LatticeLSH` is omitted: it only depends
            // on the data offset, which is zero here, and it is a construction
            // time cost rather than a per hash one.
            projection_bias.reserve( num_projections );
            for ( size_t i = 0; i < num_projections; i++ ) {
                projection_bias.push_back( panna::sample_random_01() );
            }

            projections.resize( num_projections, 0.0f );
        }

        size_t matrix_bytes() const {
            return matrix.size() * sizeof( float );
        }

        void hash( EuclideanPoints::PointHandle point, std::vector<Value>& output ) {
            output.resize( repetitions );
            const size_t num_projections = repetitions * K * LATTICE_DIMENSIONS;

            // Unlike the Hadamard based variant we need no zero padded copy of
            // the point: the dot products can read it in place.
            for ( size_t i = 0; i < num_projections; i++ ) {
                const EuclideanPointHandle row {
                    .dimensions = dimensions,
                    .vector = matrix.data() + i * dimensions
                };
                projections[i] = panna::dot_product( row, point );
            }

            // From here on this is a verbatim copy of the tail of
            // `LatticeLSH::hash`, so that only the projections differ.
            const float inv_scaling_factor = 1.0f / scaling_factor;
            const float* prjs = projections.data();
            const float* bias = projection_bias.data();
#ifdef __AVX2__
            const __m256 inv_scale = _mm256_set1_ps( inv_scaling_factor );
#endif
            for ( size_t rep = 0; rep < repetitions; rep++ ) {
                Value cur;
                const size_t rep_base = rep * LATTICE_DIMENSIONS * K;
                for ( size_t concat = 0; concat < K; concat++ ) {
                    const size_t concat_base = rep_base + concat * LATTICE_DIMENSIONS;
#ifdef __AVX2__
                    const __m256 prj = _mm256_fmadd_ps( _mm256_loadu_ps( prjs + concat_base ),
                                                        inv_scale,
                                                        _mm256_loadu_ps( bias + concat_base ) );
#else
                    float prj[LATTICE_DIMENSIONS];
                    for ( size_t i = 0; i < LATTICE_DIMENSIONS; i++ ) {
                        const size_t idx = concat_base + i;
                        prj[i] = prjs[idx] * inv_scaling_factor + bias[idx];
                    }
#endif
                    cur.set( concat, panna::decode_e8_code( prj ) );
                }
                output[rep] = cur;
            }
        }
    };

    //! How many of the `repetitions` hash values are distinct. A degenerate
    //! hasher would return the same value everywhere.
    size_t count_distinct( std::vector<Value> values ) {
        std::sort( values.begin(), values.end() );
        return static_cast<size_t>( std::distance( values.begin(),
                                                   std::unique( values.begin(), values.end() ) ) );
    }

    bool parse_positive( const char* text, size_t& out ) {
        errno = 0;
        char* end = nullptr;
        const unsigned long parsed = std::strtoul( text, &end, 10 );
        if ( errno != 0 || end == text || *end != '\0' || parsed == 0 ) {
            return false;
        }
        out = static_cast<size_t>( parsed );
        return true;
    }

} // namespace

int main( int argc, char** argv ) {
    size_t dimensions = 128;
    size_t repetitions = 100;

    if ( argc > 3 ) {
        std::cerr << "usage: " << argv[0] << " [dimensions] [repetitions]\n";
        return 1;
    }
    if ( argc > 1 && !parse_positive( argv[1], dimensions ) ) {
        std::cerr << "dimensions must be a positive integer, got '" << argv[1] << "'\n";
        return 1;
    }
    if ( argc > 2 && !parse_positive( argv[2], repetitions ) ) {
        std::cerr << "repetitions must be a positive integer, got '" << argv[2] << "'\n";
        return 1;
    }

    panna::seed_global_rng( SEED );

    const size_t num_projections = repetitions * K * LATTICE_DIMENSIONS;
    const size_t fht_size = size_t( 1 )
                            << panna::ceil_log( std::max( dimensions, num_projections ) );

    EuclideanPoints points( dimensions );
    for ( size_t i = 0; i < NUM_POINTS; i++ ) {
        points.push_back_random();
    }

    const std::vector<float> zero_offset( dimensions, 0.0f );
    ReferenceHasher fht_hasher( zero_offset, 1.0f, dimensions, repetitions );
    DenseLatticeLSH dense_hasher( 1.0f, dimensions, repetitions );

    std::cout << "K                     " << +K << "\n"
              << "dimensions            " << dimensions << "\n"
              << "repetitions           " << repetitions << "\n"
              << "projections per point " << num_projections << "\n"
              << "FHT buffer (floats)   " << fht_size << " (" << fht_size * sizeof( float )
              << " bytes)\n"
              << "dense matrix          " << dense_hasher.matrix_bytes() << " bytes\n"
              << std::endl;

    std::vector<Value> fht_hashes;
    std::vector<Value> dense_hashes;
    fht_hasher.hash( points[0], fht_hashes );
    dense_hasher.hash( points[0], dense_hashes );

    // The two hashers draw independent randomness, so their values differ; what
    // matters is that neither collapses to a constant.
    std::cout << "sanity, point 0:\n"
              << "  RandomDotProducts  first=" << fht_hashes.at( 0 ) << " distinct="
              << count_distinct( fht_hashes ) << "/" << fht_hashes.size() << "\n"
              << "  dense Gaussian     first=" << dense_hashes.at( 0 ) << " distinct="
              << count_distinct( dense_hashes ) << "/" << dense_hashes.size() << "\n"
              << std::endl;

    ankerl::nanobench::Bench bench;
    bench.title( "LatticeLSH::hash, dimensions=" + std::to_string( dimensions ) +
                 " repetitions=" + std::to_string( repetitions ) )
        .unit( "hash" )
        .relative( true )
        .minEpochIterations( 100 );

    size_t point_idx = 0;
    bench.run( "LatticeLSH (RandomDotProducts)", [&] {
        fht_hasher.hash( points[point_idx], fht_hashes );
        point_idx = ( point_idx + 1 ) % points.size();
        ankerl::nanobench::doNotOptimizeAway( fht_hashes );
    } );

    point_idx = 0;
    bench.run( "LatticeLSH (dense Gaussian)", [&] {
        dense_hasher.hash( points[point_idx], dense_hashes );
        point_idx = ( point_idx + 1 ) % points.size();
        ankerl::nanobench::doNotOptimizeAway( dense_hashes );
    } );

    return 0;
}
