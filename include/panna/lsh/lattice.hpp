#pragma once

#include <algorithm>
#include <cmath>
#include <numeric>
#include <omp.h>
#include <random>
#include <vector>

#include "panna/data.hpp"
#include "panna/distance.hpp"
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
        Dataset random_vectors; // TODO: remove
        RandomDotProducts random_dots;
        std::vector<float> offsets;
        // the corrections to apply to projections so that
        // they behave like the input vector was first offset and scaled
        std::vector<float> corrections;
        // precomputed (offset - correction) term used in the hot hash loop
        std::vector<float> projection_bias;
        // scratch space
        std::vector<std::vector<float>> tl_scratch;

        std::array<float, LATTICE_DIMENSIONS> project( typename Dataset::PointHandle point,
                                                       size_t concatenation,
                                                       size_t repetition ) const {
            std::array<float, LATTICE_DIMENSIONS> out;
            const size_t start = repetition * LATTICE_DIMENSIONS * K + concatenation * LATTICE_DIMENSIONS;
            const size_t end = start + LATTICE_DIMENSIONS;
            expect(end <= random_vectors.size());
            for (size_t i=0; i<LATTICE_DIMENSIONS; i++) {
                out[i] = dot_product(point, random_vectors[start+i]) / scaling_factor - corrections.at(start+i) + offsets.at(start+i);
            }
            // std::cout << "[ ";
            // for(size_t i=0; i<LATTICE_DIMENSIONS; i++) {
            //     std::cout << out[i] << " ";
            // }
            // std::cout << "]\n";
            return out;
        }

        std::array<float, LATTICE_DIMENSIONS> project( const std::vector<float> & projections,
                                                       size_t concatenation,
                                                       size_t repetition ) const {
            std::array<float, LATTICE_DIMENSIONS> out;
            const size_t start = repetition * LATTICE_DIMENSIONS * K + concatenation * LATTICE_DIMENSIONS;
            const size_t end = start + LATTICE_DIMENSIONS;
            expect(end <= random_vectors.size());
            for (size_t i=0; i<LATTICE_DIMENSIONS; i++) {
                out.at(i) = projections.at(start + i) / scaling_factor - corrections.at( start + i ) +
                         offsets.at( start + i );
            }
            // std::cout << "[ ";
            // for(size_t i=0; i<LATTICE_DIMENSIONS; i++) {
            //     std::cout << out[i] << " ";
            // }
            // std::cout << "]\n";
            return out;
        }


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
            random_vectors( dimensions ),
            random_dots( std::max( dimensions, repetitions * K * LATTICE_DIMENSIONS ) ),
            corrections(),
            projection_bias() {
            for ( size_t vec_idx = 0; vec_idx < repetitions * K * LATTICE_DIMENSIONS; vec_idx++ ) {
                std::vector<float> dir = sample_random_normal_vector( dimensions, rng );
                rescale( dir, 1.0 / std::sqrt( LATTICE_DIMENSIONS ) );
                random_vectors.push_back( dir.begin(), dir.end() );
                float offset = sample_random_01(rng);
                float correction = dot_product(dir, data_offset) / scaling_factor;
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
                random_vectors,
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
            // use the projections
            for ( size_t rep = 0; rep < repetitions; rep++ ) {
                Value cur;
                const size_t rep_base = rep * LATTICE_DIMENSIONS * K;
                for ( size_t concat = 0; concat < K; concat++ ) {
                    std::array<float, LATTICE_DIMENSIONS> prj;
                    const size_t concat_base = rep_base + concat * LATTICE_DIMENSIONS;
                    for ( size_t i = 0; i < LATTICE_DIMENSIONS; i++ ) {
                        const size_t idx = concat_base + i;
                        prj.at(i) = scratch.at(idx) * inv_scaling_factor + projection_bias.at(idx);
                    }
                    auto decoded = decode_e8(prj);
                    int64_t code = to_int64(to_integer_coords(decoded));
                    cur.set( concat, code );
                }
                output.at(rep) = cur;
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
        static constexpr size_t FIT_MIN_SAMPLE_SIZE = 2048;

        Dataset sample_points( const Dataset& points,
                               std::vector<size_t>* sampled_original_indices = nullptr ) const {
            const size_t n = points.size();
            if ( n == 0 ) {
                return Dataset( dimensions );
            }

            size_t target = static_cast<size_t>( n * FIT_SAMPLE_RATIO );
            target = std::max<size_t>( 1, std::max( target, FIT_MIN_SAMPLE_SIZE ) );
            target = std::min( target, n );

            if ( target == n ) {
                if ( sampled_original_indices ) {
                    sampled_original_indices->resize( n );
                    std::iota( sampled_original_indices->begin(), sampled_original_indices->end(), 0 );
                }
                return points;
            }

            std::vector<size_t> order( n );
            std::iota( order.begin(), order.end(), 0 );
            auto& rng = get_global_rng();
            std::shuffle( order.begin(), order.end(), rng );

            Dataset sampled( dimensions );
            std::vector<float> scratch( dimensions );
            if ( sampled_original_indices ) {
                sampled_original_indices->clear();
                sampled_original_indices->reserve( target );
            }
            for ( size_t i = 0; i < target; i++ ) {
                size_t original_idx = order.at(i);
                auto point = points[original_idx];
                point.into_vec( scratch );
                sampled.push_back( scratch.begin(), scratch.end() );
                if ( sampled_original_indices ) {
                    sampled_original_indices->push_back( original_idx );
                }
            }
            return sampled;
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

        void fit( Dataset& points ) {
            if ( scaling_factor != 0.0 ) {
                return;
            }
            Dataset fit_points = sample_points( points );
            if ( fit_points.size() == 0 ) {
                throw std::invalid_argument( "cannot fit hash builder on an empty dataset" );
            }
            const size_t fit_n = fit_points.size();
            offset = mean_point(fit_points);
            const float diameter = approximate_diameter<Distance>(fit_points);
            const size_t sample_repetitions = 4;
            LOG_INFO("diameter", diameter);

            auto compute_avg_collisions = [&](float scale) -> float {
                std::vector<PrefixMap<typename Output::Value>> pmaps(sample_repetitions);
                Output hasher(offset, scale, dimensions, sample_repetitions);
                PrefixMap<typename Output::Value>::populate_from(pmaps, fit_points, hasher);

                size_t collisions = 0;
                for (auto& pmap : pmaps) {
                    PairPrefixMapCursorNew<typename Output::Value> cursor =
                        pmap.create_pair_cursor_new(hasher.get_concatenations(), std::nullopt);
                    collisions += cursor.total_collisions();
                }
                return static_cast<float>(collisions) / pmaps.size();
            };

            // TODO: make these configurable to handle different scenarios
            const float threshold_low = fit_n / 2.0;
            const float threshold_high = fit_n * 2.0;
            LOG_INFO("threshold-low", threshold_low, "threshold_high", threshold_high);

            float low=0.0, high=diameter;
            const size_t MAX_ITER = 40;
            for(size_t iter=0; iter<MAX_ITER; iter++) {
                float scale = (low+high) / 2.0;
                float avg_collisions = compute_avg_collisions(scale);
                LOG_INFO("scale", scale, "avg-collisions", avg_collisions);
                if (threshold_low <= avg_collisions && avg_collisions <= threshold_high) {
                    scaling_factor = scale;
                    break;
                } else if (avg_collisions < threshold_low) {
                    low = scale;
                } else {
                    high = scale;
                }
            }
            LOG_INFO("scaling-factor", scaling_factor);
        }

        void fit( Dataset& points, std::function<uint32_t( uint32_t )> group_fun ) {
            const float old_scaling_factor = scaling_factor;
            scaling_factor = 0.0;
            std::vector<size_t> sampled_original_indices;
            Dataset fit_points = sample_points( points, &sampled_original_indices );
            if ( fit_points.size() == 0 ) {
                throw std::invalid_argument( "cannot fit hash builder on an empty dataset" );
            }
            const size_t fit_n = fit_points.size();
            offset = mean_point( fit_points );
            const float diameter = approximate_diameter<Distance>( fit_points );
            const size_t sample_repetitions = 4;
            LOG_INFO( "diameter", diameter );

            std::vector<uint32_t> sampled_groups( fit_n );
            for ( size_t i = 0; i < fit_n; i++ ) {
                sampled_groups.at(i) = group_fun( sampled_original_indices.at(i) );
            }

            auto compute_avg_collisions = [&]( float scale ) -> float {
                std::vector<PrefixMap<typename Output::Value>> pmaps( sample_repetitions );
                Output hasher( offset, scale, dimensions, sample_repetitions );
                PrefixMap<typename Output::Value>::populate_from( pmaps, fit_points, hasher );

                size_t collisions = 0;
                for ( auto& pmap : pmaps ) {
                    auto cursor = pmap.create_pair_cursor_grouped(
                        hasher.get_concatenations(),
                        std::nullopt,
                        [&]( uint32_t x ) { return sampled_groups.at(x); } );
                    collisions += cursor.total_collisions();
                }
                return static_cast<float>( collisions ) / pmaps.size();
            };

            // TODO: make these configurable to handle different scenarios
            const float threshold_low = std::sqrt(fit_n)/2;
            const float threshold_high = fit_n * 2.0;
            LOG_INFO( "threshold-low", threshold_low, "threshold_high", threshold_high );

            float low = 2 * old_scaling_factor;
            float high = std::max( diameter, low * 1.01f );
            expect( low <= high );
            const size_t MAX_ITER = 40;
            bool found = false;
            for ( size_t iter = 0; iter < MAX_ITER; iter++ ) {
                float scale = ( low + high ) / 2.0;
                float avg_collisions = compute_avg_collisions( scale );
                LOG_INFO( "scale", scale, "avg-collisions", avg_collisions );
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
                scaling_factor = low;
            }
            LOG_INFO( "scaling-factor", scaling_factor );
            expect(scaling_factor > old_scaling_factor);
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
