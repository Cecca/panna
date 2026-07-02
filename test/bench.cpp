#include "panna/data.hpp"
#define ANKERL_NANOBENCH_IMPLEMENT
#include <nanobench.h>
#include "panna/rand.hpp"
#include "panna/linalg.hpp"
#include "panna/distance.hpp"
#include "panna/lsh/lattice.hpp"

int main() {
    using namespace panna;
    double dims = 980;
    std::vector<float> a = sample_random_normal_vector(dims);
    std::vector<float> b = sample_random_normal_vector(dims);
    NormedPoints pts(dims);
    pts.push_back(a.begin(), a.end());
    pts.push_back(b.begin(), b.end());

    ankerl::nanobench::Bench().run("euclidean naive", [&] {
        float d = euclidean_naive( a.data(), b.data(), dims );
        ankerl::nanobench::doNotOptimizeAway(d);
    });
    ankerl::nanobench::Bench().run("euclidean avx2", [&] {
        float d = euclidean_avx2( a.data(), b.data(), dims );
        ankerl::nanobench::doNotOptimizeAway(d);
    });
    ankerl::nanobench::Bench().run("euclidean normed points", [&] {
        float d = euclidean( pts[0], pts[1] );
        ankerl::nanobench::doNotOptimizeAway(d);
    });

    // cycle through many random inputs so the decoder cannot be constant-folded
    const size_t num_inputs = 1024;
    std::vector<std::array<float, 8>> inputs( num_inputs );
    for ( auto& input : inputs ) {
        auto v = sample_random_normal_vector( 8 );
        std::copy( v.begin(), v.end(), input.begin() );
    }
    size_t input_idx = 0;
    ankerl::nanobench::Bench().minEpochIterations( 100000 ).run( "decode_e8", [&] {
        auto decoded = decode_e8( inputs[input_idx] );
        input_idx = ( input_idx + 1 ) % num_inputs;
        ankerl::nanobench::doNotOptimizeAway( decoded );
    });
    input_idx = 0;
    ankerl::nanobench::Bench().minEpochIterations( 100000 ).run( "decode_e8 to code", [&] {
        auto decoded = decode_e8( inputs[input_idx] );
        int64_t code = to_int64( to_integer_coords( decoded ) );
        input_idx = ( input_idx + 1 ) % num_inputs;
        ankerl::nanobench::doNotOptimizeAway( code );
    });
    input_idx = 0;
    ankerl::nanobench::Bench().minEpochIterations( 100000 ).run( "decode_e8_code scalar", [&] {
        int64_t code = decode_e8_code( inputs[input_idx].data() );
        input_idx = ( input_idx + 1 ) % num_inputs;
        ankerl::nanobench::doNotOptimizeAway( code );
    });
#ifdef __AVX2__
    input_idx = 0;
    ankerl::nanobench::Bench().minEpochIterations( 100000 ).run( "decode_e8_code avx2", [&] {
        int64_t code = decode_e8_code( _mm256_loadu_ps( inputs[input_idx].data() ) );
        input_idx = ( input_idx + 1 ) % num_inputs;
        ankerl::nanobench::doNotOptimizeAway( code );
    });
#endif

    const size_t hash_dims = 128;
    const size_t hash_repetitions = 100;
    EuclideanPoints hash_pts( hash_dims );
    for ( size_t i = 0; i < 8; i++ ) {
        hash_pts.push_back_random();
    }
    using HashFamily = LatticeLSH<1, EuclideanPoints, EuclideanDistance>;
    std::vector<float> hash_offset( hash_dims, 0.0f );
    HashFamily lattice_hasher( hash_offset, 1.0f, hash_dims, hash_repetitions );
    std::vector<HashFamily::Value> hashes;
    size_t point_idx = 0;
    ankerl::nanobench::Bench().minEpochIterations( 10000 ).run( "LatticeLSH::hash", [&] {
        lattice_hasher.hash( hash_pts[point_idx], hashes );
        point_idx = ( point_idx + 1 ) % hash_pts.size();
        ankerl::nanobench::doNotOptimizeAway( hashes );
    });
}
