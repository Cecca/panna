#pragma once

#include <atomic>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "ffht/fht_header_only.h"
#include "panna/data.hpp"

#if defined( __AVX2__ ) || defined( __AVX__ )
    #include <immintrin.h>
#endif

namespace panna {

    template <typename T>
    static T add( T &a, T &b );

    template<>
    std::vector<float> add(std::vector<float> &a, std::vector<float>&b) {
        expect(a.size() == b.size());
        std::vector<float> out(a.size());
        for (size_t i=0; i<a.size(); i++) {
            out[i] = a[i] + b[i];
        }
        return out;
    }

    

    template <typename T>
    static float dot_product( T a, T b );

    template<>
    float dot_product( std::vector<float> a, std::vector<float> b ) {
        assert( a.size() == b.size() );
        float sum = 0.0;
        for ( size_t i = 0; i < a.size(); i++ ) {
            sum += a[i] * b[i];
        }
        return sum;
    }

    template<>
    float dot_product( EuclideanPointHandle a, EuclideanPointHandle b ) {
        expect( a.dimensions == b.dimensions );
        float sum = 0.0;
        for ( size_t i = 0; i < a.dimensions; i++ ) {
            sum += a.vector[i] * b.vector[i];
        }
        return sum;
    }

#ifdef __AVX2__
    inline static int16_t reduce_sum( __m256i values ) {
        const static unsigned int VALUES_PER_CHUNK = 16;
        alignas( 32 ) int16_t stored[VALUES_PER_CHUNK];
        _mm256_store_si256( (__m256i*)stored, values );
        int16_t ret = 0;
        for ( unsigned i = 0; i < VALUES_PER_CHUNK; i++ ) {
            ret += stored[i];
        }
        return ret;
    }

    inline static int16_t dot_product_chunks16_avx2( UnitNormPointHandle lhs,
                                                     UnitNormPointHandle rhs ) {
        assert( lhs.num_chunks == rhs.num_chunks );
        __m256i res =
            _mm256_mulhrs_epi16( _mm256_load_si256( (__m256i*)lhs.chunks[0].chunk.data() ),
                                 _mm256_load_si256( (__m256i*)rhs.chunks[0].chunk.data() ) );
        for ( size_t i = 1; i < lhs.num_chunks; i += 1 ) {
            __m256i tmp =
                _mm256_mulhrs_epi16( _mm256_load_si256( (__m256i*)lhs.chunks[i].chunk.data() ),
                                     _mm256_load_si256( (__m256i*)rhs.chunks[i].chunk.data() ) );
            res = _mm256_add_epi16( res, tmp );
        }
        return reduce_sum( res );
    }
#endif

    inline static int16_t dot_product_chunks16_simple( UnitNormPointHandle lhs,
                                                       UnitNormPointHandle rhs ) {
        assert( lhs.num_chunks == rhs.num_chunks );
        const static unsigned int VALUES_PER_CHUNK = 16;

        int16_t res = 0;
        for ( size_t chunk_idx = 0; chunk_idx < lhs.num_chunks; chunk_idx++ ) {
            for ( size_t i = 0; i < VALUES_PER_CHUNK; i++ ) {
                int32_t precise = static_cast<int32_t>( lhs.chunks[chunk_idx].chunk[i] ) *
                                  static_cast<int32_t>( rhs.chunks[chunk_idx].chunk[i] );
                res += static_cast<int16_t>( ( ( precise >> 14 ) + 1 ) >> 1 );
            }
        }
        return res;
    }

    static inline int16_t dot_product_chunks16( UnitNormPointHandle lhs, UnitNormPointHandle rhs ) {
#ifdef __AVX2__
        return dot_product_chunks16_avx2( lhs, rhs );
#else
    #pragma message( \
        "there is no AVX instruction set on this machine, performance is severely limited" )
        return dot_product_chunks16_simple( lhs, rhs );
#endif
    }

    template <>
    float dot_product( UnitNormPointHandle a, UnitNormPointHandle b ) {
        return from_16bit_fixed_point( dot_product_chunks16( a, b ) );
    }

    template <>
    float dot_product( NormedPointHandle a, NormedPointHandle b ) {
        float inner_dot = from_16bit_fixed_point( dot_product_chunks16( a.inner, b.inner ) );
        return std::sqrt( a.squared_norm() ) * std::sqrt( b.squared_norm() ) * inner_dot;
    }

    static void normalize(std::vector<float> & point) {
        float norm = std::sqrt(dot_product(point, point));
        if (norm == 0) {
            //throw std::runtime_error("Cannot normalize a zero vector");
            return;
        }
        for (size_t i=0; i<point.size(); i++) {
            point[i] /= norm;
        }
    }

    static void rescale(std::vector<float> & point, float factor) {
        for (size_t i=0; i<point.size(); i++) {
            point[i] *= factor;
        }
    }

    
    template <typename T>
    static float euclidean( T a, T b );

    template <>
    float euclidean( NormedPointHandle a, NormedPointHandle b ) {
        float dot = dot_product( a, b );
        return std::sqrt( a.squared_norm() + b.squared_norm() - 2 * dot );
    }

    template <>
    float euclidean( EuclideanPointHandle a, EuclideanPointHandle b ) {
        float d = 0;
        for (size_t i=0; i<a.dimensions; i++) {
            float diff = a.vector[i] - b.vector[i];
            d += diff*diff;
        }
        return std::sqrt(d);
    }

    template <>
    float euclidean(std::vector<float> a, std::vector<float> b) {
        float d = 0;
        for (size_t i=0; i<a.size(); i++) {
            float diff = a[i] - b[i];
            d += diff*diff;
        }
        return std::sqrt(d);
    }

    template <size_t D>
    float euclidean(std::array<float, D> a, std::array<float, D> b) {
        float d = 0;
        for (size_t i=0; i<a.size(); i++) {
            float diff = a[i] - b[i];
            d += diff*diff;
        }
        return std::sqrt(d);
    }

    template <size_t D>
    float euclidean(std::array<float, D> a, std::array<long, D> b) {
        float d = 0;
        for (size_t i=0; i<a.size(); i++) {
            float diff = a[i] - ((float) b[i]);
            d += diff*diff;
        }
        return std::sqrt(d);
    }

    constexpr static unsigned int ceil_log( unsigned int value ) {
        unsigned int log = 0;
        unsigned int power_of_two = 1;
        while ( power_of_two < value ) {
            log++;
            power_of_two *= 2;
        }
        return log;
    }

    // Perform the dot product with many random vetors at once using the
    // Fast Hadamard Transform
    template <uint8_t ROTATIONS = 3>
    class RandomDotProducts {
        size_t num_products;
        size_t log_num_products;
        std::vector<float> random_signs;

    public:
        RandomDotProducts() {}

        RandomDotProducts( size_t num_products ):
            num_products( num_products ), log_num_products( ceil_log( num_products ) ) {

            int random_signs_len = ROTATIONS * ( 1 << log_num_products );
            random_signs.reserve( random_signs_len );

            std::uniform_int_distribution<int8_t> sign_distribution( 0, 1 );
            auto& generator = get_global_rng();
            for ( int i = 0; i < random_signs_len; i++ ) {
                random_signs.push_back( sign_distribution( generator ) * 2 - 1 );
            }
        }

        template <typename Archive>
        void serialize( Archive& ar ) {
            ar( num_products, log_num_products, random_signs );
        }

        friend bool operator==( const RandomDotProducts<ROTATIONS>& a,
                                const RandomDotProducts<ROTATIONS>& b ) {
            return a.num_products == b.num_products && a.log_num_products == b.log_num_products &&
                   a.random_signs == b.random_signs;
        }
        std::vector<float> allocate_scratch() const {
            std::vector<float> scratch;
            scratch.resize( 1 << log_num_products );
            return scratch;
        }

        void compute( std::vector<float>& in_out ) const {
            for ( uint8_t rotation = 0; rotation < ROTATIONS; rotation++ ) {
                // Multiply by a diagonal +-1 matrix.
                size_t base_idx = rotation * ( 1 << log_num_products );
                for ( size_t i = 0; i < ( 1 << log_num_products ); i++ ) {
                    // OPTIMIZE use simd, this takes half as much time as the fht transform below
                    in_out[i] *= random_signs[base_idx + i];
                }
                // Apply the fast hadamard transform
                fht( in_out.data(), log_num_products );
            }
        }
    };

} // namespace panna
