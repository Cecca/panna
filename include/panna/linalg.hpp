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

    // Overload for std::vector<float> taking const references (avoids copies)
    // and using AVX2 when available.
    static float dot_product( const std::vector<float>& a, const std::vector<float>& b ) {
        assert( a.size() == b.size() );
#ifdef __AVX2__
        const size_t n = a.size();
        const size_t n8 = n & ~(size_t)7; // round down to multiple of 8
        __m256 acc = _mm256_setzero_ps();
        for ( size_t i = 0; i < n8; i += 8 ) {
            __m256 va = _mm256_loadu_ps( &a[i] );
            __m256 vb = _mm256_loadu_ps( &b[i] );
            acc = _mm256_fmadd_ps( va, vb, acc );
        }
        // Horizontal sum of 8 floats
        __m128 hi = _mm256_extractf128_ps( acc, 1 );
        __m128 lo = _mm256_castps256_ps128( acc );
        __m128 sum4 = _mm_add_ps( lo, hi );
        sum4 = _mm_hadd_ps( sum4, sum4 );
        sum4 = _mm_hadd_ps( sum4, sum4 );
        float sum = _mm_cvtss_f32( sum4 );
        // Scalar tail for remaining elements
        for ( size_t i = n8; i < n; i++ ) {
            sum += a[i] * b[i];
        }
        return sum;
#else
        float sum = 0.0f;
        for ( size_t i = 0; i < a.size(); i++ ) {
            sum += a[i] * b[i];
        }
        return sum;
#endif
    }

    template<>
    float dot_product( std::vector<float> a, std::vector<float> b ) {
        // Forward to the const-ref overload to avoid duplicating logic
        return dot_product( static_cast<const std::vector<float>&>(a),
                            static_cast<const std::vector<float>&>(b) );
    }

    template<>
    float dot_product( EuclideanPointHandle a, EuclideanPointHandle b ) {
        expect( a.dimensions == b.dimensions );
#ifdef __AVX2__
        // Process 8 floats per iteration using aligned loads.
        // Stride is padded to multiple of 8 and padding is zero,
        // so we can iterate over the full stride without a scalar tail.
        __m256 acc = _mm256_setzero_ps();
        size_t n = a.stride;
        for ( size_t i = 0; i < n; i += 8 ) {
            __m256 va = _mm256_load_ps( a.vector + i );
            __m256 vb = _mm256_load_ps( b.vector + i );
            acc = _mm256_fmadd_ps( va, vb, acc );
        }
        // Horizontal sum of 8 floats
        __m128 hi = _mm256_extractf128_ps( acc, 1 );
        __m128 lo = _mm256_castps256_ps128( acc );
        __m128 sum4 = _mm_add_ps( lo, hi );
        sum4 = _mm_hadd_ps( sum4, sum4 );
        sum4 = _mm_hadd_ps( sum4, sum4 );
        return _mm_cvtss_f32( sum4 );
#else
        float sum = 0.0f;
        for ( size_t i = 0; i < a.dimensions; i++ ) {
            sum += a.vector[i] * b.vector[i];
        }
        return sum;
#endif
    }

#ifdef __AVX2__
    // Horizontal sum of 8 x int32 lanes into a single int32.
    inline static int32_t reduce_sum_i32( __m256i v ) {
        // v = [a0 a1 a2 a3 | a4 a5 a6 a7]
        __m128i lo = _mm256_castsi256_si128( v );
        __m128i hi = _mm256_extracti128_si256( v, 1 );
        __m128i sum = _mm_add_epi32( lo, hi );       // [a0+a4, a1+a5, a2+a6, a3+a7]
        sum = _mm_hadd_epi32( sum, sum );             // [s01, s23, s01, s23]
        sum = _mm_hadd_epi32( sum, sum );             // [s0123, ...]
        return _mm_cvtsi128_si32( sum );
    }

    // Dot product using _mm256_madd_epi16: multiplies int16 pairs and sums
    // adjacent products into int32, accumulating in int32 throughout.
    // This avoids the int16 overflow of the old mulhrs approach and gives
    // full precision (the only error is the original Q15 quantization).
    // Result is in Q30 format (two Q15 values multiplied).
    inline static int32_t dot_product_chunks_avx2( UnitNormPointHandle lhs,
                                                   UnitNormPointHandle rhs ) {
        assert( lhs.num_chunks == rhs.num_chunks );
        __m256i acc = _mm256_madd_epi16(
            _mm256_load_si256( (__m256i*)lhs.chunks[0].chunk.data() ),
            _mm256_load_si256( (__m256i*)rhs.chunks[0].chunk.data() ) );
        for ( size_t i = 1; i < lhs.num_chunks; i++ ) {
            __m256i prod = _mm256_madd_epi16(
                _mm256_load_si256( (__m256i*)lhs.chunks[i].chunk.data() ),
                _mm256_load_si256( (__m256i*)rhs.chunks[i].chunk.data() ) );
            acc = _mm256_add_epi32( acc, prod );
        }
        return reduce_sum_i32( acc );
    }
#endif

    // Scalar fallback: accumulate int16*int16 products in int32.
    // Result is in Q30 format.
    inline static int32_t dot_product_chunks_simple( UnitNormPointHandle lhs,
                                                     UnitNormPointHandle rhs ) {
        assert( lhs.num_chunks == rhs.num_chunks );
        const static unsigned int VALUES_PER_CHUNK = 16;
        int32_t acc = 0;
        for ( size_t chunk_idx = 0; chunk_idx < lhs.num_chunks; chunk_idx++ ) {
            for ( size_t i = 0; i < VALUES_PER_CHUNK; i++ ) {
                acc += static_cast<int32_t>( lhs.chunks[chunk_idx].chunk[i] ) *
                       static_cast<int32_t>( rhs.chunks[chunk_idx].chunk[i] );
            }
        }
        return acc;
    }

    // Returns the dot product of the unit-norm vectors in Q30 fixed-point.
    static inline int32_t dot_product_q30( UnitNormPointHandle lhs, UnitNormPointHandle rhs ) {
#ifdef __AVX2__
        return dot_product_chunks_avx2( lhs, rhs );
#else
    #pragma message( \
        "there is no AVX instruction set on this machine, performance is severely limited" )
        return dot_product_chunks_simple( lhs, rhs );
#endif
    }

    // Convert Q30 fixed-point to float (divide by 2^30).
    static inline float from_q30( int32_t val ) {
        return static_cast<float>( val ) / static_cast<float>( 1 << 30 );
    }

    template <>
    float dot_product( UnitNormPointHandle a, UnitNormPointHandle b ) {
        return from_q30( dot_product_q30( a, b ) );
    }

    template <>
    float dot_product( NormedPointHandle a, NormedPointHandle b ) {
        float inner_dot = from_q30( dot_product_q30( a.inner, b.inner ) );
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
        // Use the numerically stable formulation:
        //   ||a-b||² = (||a|| - ||b||)² + 2·||a||·||b||·(1 - cos(a,b))
        // This avoids catastrophic cancellation in (||a||² + ||b||² - 2·dot)
        // when points are close, and is guaranteed non-negative.
        float cos_ab = from_q30( dot_product_q30( a.inner, b.inner ) );
        // Clamp cosine to [-1, 1] to account for Q15 quantization rounding
        cos_ab = std::min( 1.0f, std::max( -1.0f, cos_ab ) );
        float norm_a = std::sqrt( a.squared_norm() );
        float norm_b = std::sqrt( b.squared_norm() );
        float norm_diff = norm_a - norm_b;
        float angular = 2.0f * norm_a * norm_b * ( 1.0f - cos_ab );
        return std::sqrt( norm_diff * norm_diff + angular );
    }

    template <>
    float euclidean( EuclideanPointHandle a, EuclideanPointHandle b ) {
        expect( a.dimensions == b.dimensions );
#ifdef __AVX2__
        // Accumulate in double precision to avoid float32 rounding errors
        // that can reorder nearly-equal edges and change the MST topology.
        __m256d acc = _mm256_setzero_pd();
        size_t n = a.stride;
        for ( size_t i = 0; i < n; i += 8 ) {
            __m256 va = _mm256_load_ps( a.vector + i );
            __m256 vb = _mm256_load_ps( b.vector + i );
            __m256 diff = _mm256_sub_ps( va, vb );
            // Convert lower 4 floats to double, square, accumulate
            __m128 lo_f = _mm256_castps256_ps128( diff );
            __m256d lo_d = _mm256_cvtps_pd( lo_f );
            acc = _mm256_fmadd_pd( lo_d, lo_d, acc );
            // Convert upper 4 floats to double, square, accumulate
            __m128 hi_f = _mm256_extractf128_ps( diff, 1 );
            __m256d hi_d = _mm256_cvtps_pd( hi_f );
            acc = _mm256_fmadd_pd( hi_d, hi_d, acc );
        }
        // Horizontal sum of 4 doubles
        __m128d hi2 = _mm256_extractf128_pd( acc, 1 );
        __m128d lo2 = _mm256_castpd256_pd128( acc );
        __m128d sum2 = _mm_add_pd( lo2, hi2 );
        __m128d hi1 = _mm_unpackhi_pd( sum2, sum2 );
        __m128d sum1 = _mm_add_sd( sum2, hi1 );
        return static_cast<float>( std::sqrt( _mm_cvtsd_f64( sum1 ) ) );
#else
        double d = 0.0;
        for (size_t i=0; i<a.dimensions; i++) {
            double diff = static_cast<double>(a.vector[i]) - static_cast<double>(b.vector[i]);
            d += diff*diff;
        }
        return static_cast<float>(std::sqrt(d));
#endif
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
