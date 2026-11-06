#pragma once

#include <random>
#include <unordered_set>
#include <vector>

namespace panna {

    static std::mt19937_64& get_global_rng() {
        static std::mt19937_64 rng;
        return rng;
    }

    //! The normal distribution used with the global RNG. It is shared so that
    //! `seed_global_rng` can reset it: `std::normal_distribution` may cache a
    //! second sample from the previous draw, which would otherwise make the
    //! draws after seeding depend on the draws before it.
    static std::normal_distribution<float>& get_global_normal_distribution() {
        static std::normal_distribution<float> normal( 0, 1 );
        return normal;
    }

    static void seed_global_rng( const uint64_t seed ) {
        auto& rng = get_global_rng();
        rng.seed( seed );
        get_global_normal_distribution().reset();
    }

    static float sample_random_01() {
        static std::uniform_real_distribution<float> unif(0, 1);
        return unif(get_global_rng());
    }

    static float sample_random_01(std::mt19937_64 & rng) {
        static std::uniform_real_distribution<float> unif(0, 1);
        return unif(rng);
    }

    static size_t sample_int(size_t min, size_t max) {
        // Not `static`: a static distribution would keep the bounds of the
        // first call, and sample out of range on later calls with other bounds.
        std::uniform_int_distribution<size_t> unif(min, max);
        return unif(get_global_rng());
    }

    static float sample_random_normal() {
        return get_global_normal_distribution()( get_global_rng() );
    }

    static float sample_random_normal(std::mt19937_64 &rng) {
        static std::normal_distribution<float> normal(0, 1);
        return normal(rng);
    }

    static std::vector<float> sample_random_normal_vector(size_t dimensions) {
        static std::normal_distribution<float> normal(0, 1);
        std::vector<float> out;
        for (size_t i=0; i<dimensions; i++) {
            out.push_back(sample_random_normal());
        }
        return out;
    }

    static std::vector<float> sample_random_normal_vector(size_t dimensions, std::mt19937_64 & rng) {
        static std::normal_distribution<float> normal(0, 1);
        std::vector<float> out;
        for (size_t i=0; i<dimensions; i++) {
            out.push_back(sample_random_normal(rng));
        }
        return out;
    }

    static std::vector<size_t> sample_k( size_t n, size_t k ) {
        // This uses Robert Floyd's algorithm: https://dl.acm.org/doi/abs/10.1145/30401.315746
        auto& gen = get_global_rng();
        std::unordered_set<size_t> s;

        for ( size_t i = n - k; i < n; ++i ) {
            std::uniform_int_distribution<size_t> dist( 0, i );
            size_t t = dist( gen );
            if ( !s.insert( t ).second )
                s.insert( i );
        }

        return std::vector<size_t>( s.begin(), s.end() );
    }

} // namespace panna
