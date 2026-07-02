#pragma once

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdlib>
#include <iterator>
#include <limits>
#include <numeric>
#include <omp.h>
#include <optional>
#include <random>
#include <vector>

#if defined( __AVX2__ ) || defined( __AVX__ )
    #include <immintrin.h>
#endif

#include "panna/data.hpp"
#include "panna/distance.hpp"
#include "panna/emst.hpp"
#include "panna/expect.hpp"
#include "panna/linalg.hpp"
#include "panna/logging.hpp"
#include "panna/lsh/values.hpp"
#include "panna/lsh/lattice_probabilities.hpp"
#include "panna/prefixmap.hpp"
#include "panna/rand.hpp"

namespace panna {

    template <size_t LATTICE_DIMENSIONS>
    static std::array<float, LATTICE_DIMENSIONS>
    decode_d8( std::array<float, LATTICE_DIMENSIONS>& x, float offset ) {
        std::array<float, LATTICE_DIMENSIONS> y;
        size_t worst_dimension = 0;
        float max_diff = 0.0;
        long rounded_sum = 0;
        // compute f(x)
        for ( size_t dim = 0; dim < LATTICE_DIMENSIONS; dim++ ) {
            float xval = x[dim] + offset;
            const long r = lroundf( xval );
            const float diff = abs( xval - r );
            rounded_sum += r;
            y[dim] = r;
            if ( diff > max_diff ) {
                worst_dimension = dim;
                max_diff = diff;
            }
        }

        if ( rounded_sum % 2 != 0 ) {
            // compute g(x), taking into account that out[worst_dimension]
            // already contains the correct rounding of x[worst_dimension]
            float frac = x[worst_dimension] + offset - y[worst_dimension];
            if ( frac > 0.5 || (-0.5 <= frac && frac < 0) ) {
                // round down instead of up
                y[worst_dimension]--;
            } else {
                // round up instead of down
                y[worst_dimension]++;
            }
        }

        for ( size_t dim = 0; dim < LATTICE_DIMENSIONS; dim++ ) {
            y[dim] -= offset;
        }

        return y;
    }

    template <size_t LATTICE_DIMENSIONS>
    static std::array<float, LATTICE_DIMENSIONS>
    decode_e8( std::array<float, LATTICE_DIMENSIONS>& x ) {
        // based on http://neilsloane.com/doc/Me83.pdf section 6
        auto snap1 = decode_d8( x, 0.0 );
        auto snap2 = decode_d8( x, -0.5 );
        if ( euclidean( x, snap1 ) < euclidean( x, snap2 ) ) {
            return snap1;
        } else {
            return snap2;
        }
    }

    template <size_t LATTICE_DIMENSIONS>
    static std::array<long, LATTICE_DIMENSIONS>
    to_integer_coords( std::array<float, LATTICE_DIMENSIONS>& y ) {
        std::array<long, LATTICE_DIMENSIONS> out;
        for ( size_t dim = 0; dim < LATTICE_DIMENSIONS; dim++ ) {
            out[dim] = static_cast<long>( y[dim] * 2 );
        }
        return out;
    }

    /// packs the given array into a 32 bit integer, taking the 4 least significant bits
    /// of each element of the array
    static int32_t to_int32( std::array<long, 8> arr ) {
        int32_t out = 0;
        for ( size_t i = 0; i < arr.size(); i++ ) {
            const int32_t bits = arr[i] & 0xF;
            out = ( out << 4 ) | bits;
        }
        return out;
    }

    static int64_t to_int64( std::array<long, 8> arr ) {
        int64_t out = 0;
        for ( size_t i = 0; i < arr.size(); i++ ) {
            const int64_t bits = arr[i] & 0xFF;
            out = ( out << 8 ) | bits;
        }
        return out;
    }

    // Decode `x` (already translated by `offset`) to the nearest point of the
    // corresponding coset of D8, producing the squared distance and the packed
    // integer code directly. `coset_one` is 1 for the D8+1/2 coset (offset
    // -0.5), where the decoded coordinates are r + 1/2, so the doubled integer
    // coordinates are 2r + 1.
    inline static void decode_d8_coset_scalar( const float* x,
                                               float offset,
                                               int32_t coset_one,
                                               float& dist_out,
                                               int64_t& code_out ) {
        int32_t r[8];
        float dist = 0.0f;
        float max_diff = 0.0f;
        float worst_diff = 0.0f;
        size_t worst = 0;
        int32_t rounded_sum = 0;
        for ( size_t dim = 0; dim < 8; dim++ ) {
            const float v = x[dim] + offset;
            const float rf = std::rint( v );
            const float diff = v - rf;
            const float adiff = std::fabs( diff );
            r[dim] = static_cast<int32_t>( rf );
            rounded_sum += r[dim];
            dist += diff * diff;
            if ( adiff > max_diff ) {
                max_diff = adiff;
                worst_diff = diff;
                worst = dim;
            }
        }
        if ( rounded_sum & 1 ) {
            // move the worst coordinate to its second-nearest integer: the
            // error there goes from |diff| to 1 - |diff|
            r[worst] += worst_diff >= 0.0f ? 1 : -1;
            dist += 1.0f - 2.0f * max_diff;
        }
        int64_t code = 0;
        for ( size_t dim = 0; dim < 8; dim++ ) {
            const int32_t coord = 2 * r[dim] + coset_one;
            code = ( code << 8 ) | ( coord & 0xFF );
        }
        dist_out = dist;
        code_out = code;
    }

    // Decode `x` to the nearest point of the E8 lattice, returning the packed
    // integer code (coordinates doubled so that D8+1/2 has integer
    // coordinates, low 8 bits each, first coordinate in the most significant
    // byte), equivalent to to_int64(to_integer_coords(decode_e8(x))).
    inline static int64_t decode_e8_code( const float* x ) {
        float d0, d1;
        int64_t c0, c1;
        decode_d8_coset_scalar( x, 0.0f, 0, d0, c0 );
        decode_d8_coset_scalar( x, -0.5f, 1, d1, c1 );
        return d0 < d1 ? c0 : c1;
    }

#ifdef __AVX2__
    inline static void decode_d8_coset_avx2( __m256 v,
                                             int32_t coset_one,
                                             float& dist_out,
                                             int64_t& code_out ) {
        const __m256 r = _mm256_round_ps( v, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC );
        const __m256 diff = _mm256_sub_ps( v, r );
        const __m256 adiff = _mm256_andnot_ps( _mm256_set1_ps( -0.0f ), diff );

        // squared distance to the rounded point
        const __m256 sq = _mm256_mul_ps( diff, diff );
        __m128 lo = _mm256_castps256_ps128( sq );
        __m128 hi = _mm256_extractf128_ps( sq, 1 );
        __m128 sum = _mm_add_ps( lo, hi );
        sum = _mm_add_ps( sum, _mm_movehl_ps( sum, sum ) );
        sum = _mm_add_ss( sum, _mm_shuffle_ps( sum, sum, 1 ) );
        float dist = _mm_cvtss_f32( sum );

        __m256i ri = _mm256_cvtps_epi32( r );

        // parity of the sum of the rounded coordinates: xor of their low bits
        const int lsb_mask =
            _mm256_movemask_ps( _mm256_castsi256_ps( _mm256_slli_epi32( ri, 31 ) ) );
        const int32_t odd = std::popcount( static_cast<unsigned>( lsb_mask ) ) & 1;

        // lowest lane holding the maximum |diff|
        __m256 m = adiff;
        m = _mm256_max_ps( m, _mm256_permute2f128_ps( m, m, 1 ) );
        m = _mm256_max_ps( m, _mm256_shuffle_ps( m, m, _MM_SHUFFLE( 1, 0, 3, 2 ) ) );
        m = _mm256_max_ps( m, _mm256_shuffle_ps( m, m, _MM_SHUFFLE( 2, 3, 0, 1 ) ) );
        const float max_diff = _mm_cvtss_f32( _mm256_castps256_ps128( m ) );
        const int eq_mask = _mm256_movemask_ps( _mm256_cmp_ps( adiff, m, _CMP_EQ_OQ ) );
        const int32_t worst = std::countr_zero( static_cast<unsigned>( eq_mask ) );

        // when the parity is odd, move the worst coordinate to its
        // second-nearest integer, i.e. by +-1 towards the actual value
        const __m256i toward =
            _mm256_add_epi32( _mm256_set1_epi32( 1 ),
                              _mm256_slli_epi32(
                                  _mm256_srai_epi32( _mm256_castps_si256( diff ), 31 ), 1 ) );
        const __m256i lane_ids = _mm256_setr_epi32( 0, 1, 2, 3, 4, 5, 6, 7 );
        const __m256i worst_mask = _mm256_cmpeq_epi32( lane_ids, _mm256_set1_epi32( worst ) );
        const __m256i odd_mask = _mm256_set1_epi32( -odd );
        ri = _mm256_add_epi32(
            ri, _mm256_and_si256( _mm256_and_si256( toward, worst_mask ), odd_mask ) );
        dist += static_cast<float>( odd ) * ( 1.0f - 2.0f * max_diff );

        const __m256i coords =
            _mm256_add_epi32( _mm256_slli_epi32( ri, 1 ), _mm256_set1_epi32( coset_one ) );
        // pack the low byte of each 32-bit lane, first coordinate in the most
        // significant byte of the resulting int64
        const __m256i shuf = _mm256_setr_epi8(
            /* 128-bit lane 0 (coords 0..3 -> bytes 4..7) */
            -1, -1, -1, -1, 12, 8, 4, 0, -1, -1, -1, -1, -1, -1, -1, -1,
            /* 128-bit lane 1 (coords 4..7 -> bytes 0..3) */
            12, 8, 4, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 );
        const __m256i packed = _mm256_shuffle_epi8( coords, shuf );
        const int64_t low = _mm_cvtsi128_si64( _mm256_castsi256_si128( packed ) );
        const int64_t high = _mm_cvtsi128_si64( _mm256_extracti128_si256( packed, 1 ) );

        dist_out = dist;
        code_out = low | high;
    }

    inline static int64_t decode_e8_code( __m256 x ) {
        float d0, d1;
        int64_t c0, c1;
        decode_d8_coset_avx2( x, 0, d0, c0 );
        decode_d8_coset_avx2( _mm256_add_ps( x, _mm256_set1_ps( -0.5f ) ), 1, d1, c1 );
        return d0 < d1 ? c0 : c1;
    }
#endif

    template <uint8_t K, typename Dataset, typename Distance>
    class LatticeLSHBuilder;

    template <uint8_t K, typename Dataset, typename Distance>
    class LatticeLSH {
    public:
        //! The datatype of the output
        using Value = LongLshValue<K>;
        using Builder = LatticeLSHBuilder<K, Dataset, Distance>;
        static const size_t LATTICE_DIMENSIONS = 8;

    private:

        std::vector<float> data_offset;
        float scaling_factor;
        size_t dimensions;
        size_t repetitions;
        RandomDotProducts random_dots;
        std::vector<float> offsets;
        // the corrections to apply to projections so that
        // they behave like the input vector was first offset and scaled
        std::vector<float> corrections;
        // precomputed (offset - correction) term used in the hot hash loop
        std::vector<float> projection_bias;
        // scratch space
        std::vector<std::vector<float>> tl_scratch;

    public:
        LatticeLSH() {
        }

        LatticeLSH( std::vector<float> offset,
                    float scaling_factor,
                    size_t dimensions,
                    size_t repetitions ):
            LatticeLSH( offset, scaling_factor, dimensions, repetitions, get_global_rng() ) {
        }

        LatticeLSH( std::vector<float> offset,
                    float scaling_factor,
                    size_t dimensions,
                    size_t repetitions,
                    std::mt19937_64& rng ):
            data_offset( offset ),
            scaling_factor( scaling_factor ),
            dimensions( dimensions ),
            repetitions( repetitions ),
            random_dots( std::max( dimensions, repetitions * K * LATTICE_DIMENSIONS ) ),
            corrections(),
            projection_bias() {

            const size_t num_projections = repetitions * K * LATTICE_DIMENSIONS;
            offsets.reserve( num_projections );
            corrections.reserve( num_projections );
            projection_bias.reserve( num_projections );

            // Project the data offset through the very same random directions that
            // `random_dots` applies at hash time.
            std::vector<float> offset_projection = random_dots.allocate_scratch();
            for ( size_t i = 0; i < dimensions && i < data_offset.size(); i++ ) {
                offset_projection.at( i ) = data_offset.at( i );
            }
            random_dots.compute( offset_projection, 1.0 / std::sqrt( LATTICE_DIMENSIONS ) );

            for ( size_t idx = 0; idx < num_projections; idx++ ) {
                const float offset = sample_random_01( rng );
                const float correction = offset_projection.at( idx ) / scaling_factor;
                offsets.push_back( offset );
                corrections.push_back( correction );
                projection_bias.push_back( offset - correction );
            }

            // prepare thread local scratch space
            for ( int i = 0; i < omp_get_max_threads(); i++ ) {
                tl_scratch.push_back( random_dots.allocate_scratch() );
            }
        }

        template <typename Archive>
        void serialize( Archive& ar ) {
            ar( data_offset,
                scaling_factor,
                repetitions,
                offsets,
                corrections,
                projection_bias );
        }

        static constexpr size_t get_concatenations() {
            return K;
        }

        size_t get_repetitions() const {
            return repetitions;
        }

        void hash( typename Dataset::PointHandle point, std::vector<Value>& output ) {
            auto& scratch = tl_scratch.at(omp_get_thread_num());
            point.into_vec(scratch);
            if ( scratch.size() > dimensions ) {
                std::fill( scratch.begin() + dimensions, scratch.end(), 0.0f );
            }
            output.resize( repetitions );
            // compute all projections in one go, scaling by the factor required to make
            // the hashing work
            random_dots.compute( scratch, 1.0 / std::sqrt( LATTICE_DIMENSIONS ) );
            const float inv_scaling_factor = 1.0f / scaling_factor;
            const float* projections = scratch.data();
            const float* bias = projection_bias.data();
#ifdef __AVX2__
            const __m256 inv_scale = _mm256_set1_ps( inv_scaling_factor );
#endif
            // use the projections
            for ( size_t rep = 0; rep < repetitions; rep++ ) {
                Value cur;
                const size_t rep_base = rep * LATTICE_DIMENSIONS * K;
                for ( size_t concat = 0; concat < K; concat++ ) {
                    const size_t concat_base = rep_base + concat * LATTICE_DIMENSIONS;
#ifdef __AVX2__
                    const __m256 prj =
                        _mm256_fmadd_ps( _mm256_loadu_ps( projections + concat_base ),
                                         inv_scale,
                                         _mm256_loadu_ps( bias + concat_base ) );
#else
                    float prj[LATTICE_DIMENSIONS];
                    for ( size_t i = 0; i < LATTICE_DIMENSIONS; i++ ) {
                        const size_t idx = concat_base + i;
                        prj[i] = projections[idx] * inv_scaling_factor + bias[idx];
                    }
#endif
                    cur.set( concat, decode_e8_code( prj ) );
                }
                output[rep] = cur;
            }
        }

        float collision_probability( float distance ) const {
            distance = Distance::to_euclidean(distance); // This gives the chance of applying the square root
            distance = distance / scaling_factor;
            if (distance > panna::lattice_lsh::MAX_DISTANCE) {
                return 0.0;
            }
            size_t idx = std::floor( distance / panna::lattice_lsh::DISTANCE_STEP );
            if ( idx < panna::lattice_lsh::NUM_ESTIMATES ) {
                return panna::lattice_lsh::PROBABILITIES[idx];
            } else {
                return 0;
            }
        }
    };

    template <uint8_t K, typename Dataset, typename Distance>
    class LatticeLSHBuilder {
        std::vector<float> offset;
        float scaling_factor = 0.0;
        size_t dimensions = 0;

        static constexpr float FIT_SAMPLE_RATIO = 0.2f;

        // If PANNA_LATTICE_SCALING_FACTOR is set to a positive value, fitting is
        // skipped and this value is used as the scaling factor.
        static std::optional<float> scaling_factor_override() {
            if ( const char* env = std::getenv( "PANNA_LATTICE_SCALING_FACTOR" ); env != nullptr ) {
                char* end = nullptr;
                const float parsed = std::strtof( env, &end );
                if ( end != env && *end == '\0' && parsed > 0.0f ) {
                    return parsed;
                }
            }
            return std::nullopt;
        }

        static std::vector<uint32_t> sample_fit_indices( size_t n ) {
            expect( n > 0 );
            const size_t sample_size = std::max<size_t>( 1, static_cast<size_t>( std::ceil( n * FIT_SAMPLE_RATIO ) ) );
            std::vector<uint32_t> all_indices( n );
            std::iota( all_indices.begin(), all_indices.end(), 0 );
            std::vector<uint32_t> sampled_indices;
            sampled_indices.reserve( sample_size );
            std::sample(
                all_indices.begin(),
                all_indices.end(),
                std::back_inserter( sampled_indices ),
                sample_size,
                get_global_rng() );
            return sampled_indices;
        }

        template <typename Hasher>
        static void populate_from_sample(
            std::vector<PrefixMap<typename LatticeLSH<K, Dataset, Distance>::Value>>& pmaps,
            Dataset& points,
            Hasher& hasher,
            const std::vector<uint32_t>& sampled_indices ) {
            std::vector<typename Hasher::Value> hashes;

#pragma omp parallel for private( hashes )
            for ( size_t i = 0; i < sampled_indices.size(); i++ ) {
                const auto tid = omp_get_thread_num();
                const uint32_t point_idx = sampled_indices.at( i );
                hasher.hash( points[point_idx], hashes );
                for ( size_t rep = 0; rep < pmaps.size(); rep++ ) {
                    pmaps[rep].insert( tid, point_idx, hashes.at( rep ) );
                }
            }

#pragma omp parallel for
            for ( size_t rep = 0; rep < pmaps.size(); rep++ ) {
                pmaps[rep].rebuild();
            }
        }

    public:
        using Output = LatticeLSH<K, Dataset, Distance>;

        LatticeLSHBuilder() {
        }

        LatticeLSHBuilder( size_t dimensions ):
            offset( dimensions ), scaling_factor( 0 ), dimensions( dimensions ) {
        }

        LatticeLSHBuilder( float offset, float scaling_factor, size_t dimensions ):
            offset( offset ), scaling_factor( scaling_factor ), dimensions( dimensions ) {
        }

        template <typename Archive>
        void serialize( Archive& ar ) {
            ar( offset, scaling_factor, dimensions );
        }

        void fit( const Dataset& points,
                  const float distance_upper_bound,
                  const size_t repetitions,
                  const float delta ) {
            if ( scaling_factor != 0.0 ) {
                return;
            }
            if ( points.size() == 0 ) {
                throw std::invalid_argument( "cannot fit hash builder on an empty dataset" );
            }
            if ( const auto override = scaling_factor_override() ) {
                offset = mean_point( points );
                scaling_factor = *override;
                LOG_INFO( "scaling-factor", scaling_factor, "source", "env-override" );
                return;
            }
            offset = mean_point( points );

            auto find_scale = [delta, repetitions]( float distance ) -> float {
                auto failure_probability = [repetitions]( float scaling_factor, float distance ) {
                    auto collision_probability = [&]( float distance ) -> float {
                        distance = Distance::to_euclidean(
                            distance ); // This gives the chance of applying the square root
                        distance = distance / scaling_factor;
                        if ( distance > panna::lattice_lsh::MAX_DISTANCE ) {
                            return 0.0;
                        }
                        size_t idx = std::floor( distance / panna::lattice_lsh::DISTANCE_STEP );
                        if ( idx < panna::lattice_lsh::NUM_ESTIMATES ) {
                            return panna::lattice_lsh::PROBABILITIES[idx];
                        } else {
                            return 0;
                        }
                    };

                    float cp = collision_probability( distance );
                    return std::pow( 1 - cp, repetitions );
                };

                // Find the smallest scaling factor for which the failure probability at the
                // heaviest EMST edge is below delta. The failure probability decreases
                // monotonically with the scaling factor, so first bracket the answer with an
                // exponential search, then refine the bracket with a binary search.
                const float min_scale = std::numeric_limits<float>::epsilon();
                float high = std::max( Distance::to_euclidean( distance ), min_scale );
                for ( size_t iter = 0; iter < 64 && failure_probability( high, distance ) >= delta;
                      iter++ ) {
                    high *= 2.0f;
                }
                float low = high / 2.0f;
                while ( low > min_scale && failure_probability( low, distance ) < delta ) {
                    high = low;
                    low /= 2.0f;
                }

                // invariant: failure_probability(high) < delta <= failure_probability(low)
                const size_t MAX_ITER = 64;
                for ( size_t iter = 0; iter < MAX_ITER && high - low > 1e-3f * high; iter++ ) {
                    const float mid = ( low + high ) / 2.0f;
                    if ( failure_probability( mid, distance ) < delta ) {
                        high = mid;
                    } else {
                        low = mid;
                    }
                }
                LOG_INFO( "proposed-scaling-factor", high );
                return high;
            };

            scaling_factor = find_scale( distance_upper_bound ) ;

            LOG_INFO("scaling-factor", scaling_factor);
            expect( scaling_factor > 0.0f );
        }
        void fit( const Dataset& points, const size_t repetitions, const float delta ) {
            if ( scaling_factor != 0.0 ) {
                return;
            }
            if ( points.size() == 0 ) {
                throw std::invalid_argument( "cannot fit hash builder on an empty dataset" );
            }
            if ( const auto override = scaling_factor_override() ) {
                offset = mean_point( points );
                scaling_factor = *override;
                LOG_INFO( "scaling-factor", scaling_factor, "source", "env-override" );
                return;
            }
            offset = mean_point( points );

            const size_t clustering_size = std::ceil( std::sqrt( points.size() ) );
            const auto clustering = kcenter<Distance>( points, clustering_size );
            const auto [weight, edges] = exact_emst<Dataset, Distance>( clustering.centers );
            const float clustering_upper_bound = std::max(edges.back().weight, clustering.radius);
            LOG_INFO("clustering-size", clustering_size, "heaviest-edge", edges.back().weight, "radius", clustering.radius, "upper-bound", clustering_upper_bound);

            const auto rand_tree = random_emst<Dataset, Distance>( points );
            const float random_upper_bound = rand_tree.back().weight;
            LOG_INFO( "random-upper-bound", random_upper_bound );

            auto find_scale = [delta, repetitions]( float distance ) -> float {
                auto failure_probability = [repetitions]( float scaling_factor, float distance ) {
                    auto collision_probability = [&]( float distance ) -> float {
                        distance = Distance::to_euclidean(
                            distance ); // This gives the chance of applying the square root
                        distance = distance / scaling_factor;
                        if ( distance > panna::lattice_lsh::MAX_DISTANCE ) {
                            return 0.0;
                        }
                        size_t idx = std::floor( distance / panna::lattice_lsh::DISTANCE_STEP );
                        if ( idx < panna::lattice_lsh::NUM_ESTIMATES ) {
                            return panna::lattice_lsh::PROBABILITIES[idx];
                        } else {
                            return 0;
                        }
                    };

                    float cp = collision_probability( distance );
                    return std::pow( 1 - cp, repetitions );
                };

                // Find the smallest scaling factor for which the failure probability at the
                // heaviest EMST edge is below delta. The failure probability decreases
                // monotonically with the scaling factor, so first bracket the answer with an
                // exponential search, then refine the bracket with a binary search.
                const float min_scale = std::numeric_limits<float>::epsilon();
                float high = std::max( Distance::to_euclidean( distance ), min_scale );
                for ( size_t iter = 0; iter < 64 && failure_probability( high, distance ) >= delta;
                      iter++ ) {
                    high *= 2.0f;
                }
                float low = high / 2.0f;
                while ( low > min_scale && failure_probability( low, distance ) < delta ) {
                    high = low;
                    low /= 2.0f;
                }

                // invariant: failure_probability(high) < delta <= failure_probability(low)
                const size_t MAX_ITER = 64;
                for ( size_t iter = 0; iter < MAX_ITER && high - low > 1e-3f * high; iter++ ) {
                    const float mid = ( low + high ) / 2.0f;
                    if ( failure_probability( mid, distance ) < delta ) {
                        high = mid;
                    } else {
                        low = mid;
                    }
                }
                LOG_INFO( "proposed-scaling-factor", high );
                return high;
            };

            scaling_factor = std::min( find_scale( random_upper_bound ), find_scale( clustering_upper_bound ) );

            LOG_INFO("scaling-factor", scaling_factor);
            expect( scaling_factor > 0.0f );
        }

        void fit( Dataset& points, std::function<uint32_t( uint32_t )> group_fun ) {
            const float old_scaling_factor = scaling_factor;
            scaling_factor = 0.0;
            if ( points.size() == 0 ) {
                throw std::invalid_argument( "cannot fit hash builder on an empty dataset" );
            }
            if ( const auto override = scaling_factor_override() ) {
                offset = mean_point( points );
                scaling_factor = *override;
                LOG_INFO( "scaling-factor", scaling_factor, "source", "env-override" );
                expect( scaling_factor > 0.0f );
                return;
            }
            const size_t fit_n = points.size();
            const auto sampled_indices = sample_fit_indices( fit_n );
            const size_t sampled_n = sampled_indices.size();
            offset = mean_point( points );
            const float diameter = approximate_diameter<Distance>( points );
            const size_t sample_repetitions = 4;
            LOG_INFO( "diameter", diameter );
            LOG_INFO( "fit-n", fit_n, "sampled-fit-n", sampled_n );

            auto compute_avg_collisions = [&]( float scale ) -> float {
                std::vector<PrefixMap<typename Output::Value>> pmaps( sample_repetitions );
                Output hasher( offset, scale, dimensions, sample_repetitions );
                populate_from_sample( pmaps, points, hasher, sampled_indices );

                size_t collisions = 0;
                for ( auto& pmap : pmaps ) {
                    auto cursor = pmap.create_pair_cursor_grouped(
                        hasher.get_concatenations(),
                        std::nullopt,
                        [&]( uint32_t x ) { return group_fun(x); } );
                    collisions += cursor.total_collisions();
                }
                return static_cast<float>( collisions ) / pmaps.size();
            };

            // TODO: make these configurable to handle different scenarios
            const float threshold_low = std::sqrt( sampled_n ) / 2.0;
            const float threshold_high = sampled_n * 2.0;
            LOG_INFO( "threshold-low", threshold_low, "threshold_high", threshold_high );

            float low = 2 * old_scaling_factor;
            if ( low <= 0.0f ) {
                low = std::max( diameter / 16.0f, std::numeric_limits<float>::epsilon() );
            }
            float high = std::max( diameter, low * 1.01f );
            expect( low <= high );
            const size_t MAX_ITER = 40;
            bool found = false;
            float best_scale = low;
            float best_error = std::numeric_limits<float>::infinity();
            for ( size_t iter = 0; iter < MAX_ITER; iter++ ) {
                float scale = ( low + high ) / 2.0;
                float avg_collisions = compute_avg_collisions( scale );
                LOG_INFO( "scale", scale, "avg-collisions", avg_collisions );
                const float error =
                    ( avg_collisions < threshold_low )
                        ? ( threshold_low - avg_collisions )
                        : ( avg_collisions > threshold_high ? ( avg_collisions - threshold_high ) : 0.0f );
                if ( error < best_error ) {
                    best_error = error;
                    best_scale = scale;
                }
                if ( threshold_low <= avg_collisions && avg_collisions <= threshold_high ) {
                    scaling_factor = scale;
                    found = true;
                    break;
                } else if ( avg_collisions < threshold_low ) {
                    low = scale;
                } else {
                    high = scale;
                }
            }
            if (!found) {
                scaling_factor = std::max( best_scale, std::numeric_limits<float>::epsilon() );
            }
            LOG_INFO( "scaling-factor", scaling_factor );
            expect( scaling_factor > 0.0f );
        }

        Output build( size_t repetitions ) const {
            expect( scaling_factor > 0 );
            return LatticeLSH<K, Dataset, Distance>( offset, scaling_factor, dimensions, repetitions );
        }

        std::string describe() const {
            std::stringstream sstream;
            sstream << "LatticeLSH(scaling=" << scaling_factor << ")";
            return sstream.str();
        }
    };

} // namespace panna
