#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numeric>
#include <optional>
#include <random>
#include <stdexcept>
#include <vector>

#include "panna/expect.hpp"
#include "panna/linalg.hpp"
#include "panna/lsh/values.hpp"
#include "panna/prefixmap.hpp"
#include "panna/rand.hpp"
#include "panna/logging.hpp"

namespace panna {

    // from https://en.cppreference.com/w/cpp/numeric/math/erfc
    static double normal_cdf( double x ) {
        return std::erfc( -x / std::sqrt( 2 ) ) / 2;
    }

    template <uint8_t K, typename Dataset, typename Distance>
    class E2LSHBuilder;

    template <uint8_t K, typename Dataset, typename Distance>
    class E2LSHBallSampler;

    template <uint8_t K, typename Dataset, typename Distance>
    class E2LSH {
    public:
        //! The datatype of the output
        using Value = BytewiseLshValue<K>;
        using Builder = E2LSHBuilder<K, Dataset, Distance>;

    private:
        friend class E2LSHBallSampler<K, Dataset, Distance>;

        float quantization_width;
        size_t repetitions;
        Dataset random_vectors;
        std::vector<float> offsets;

    public:
        E2LSH() {
        }

        E2LSH( float quantization_width, size_t dimensions, size_t repetitions ):
            quantization_width( quantization_width ),
            repetitions( repetitions ),
            random_vectors( dimensions ) {
            auto& rng = get_global_rng();
            std::uniform_real_distribution<float> uniform( 0.0, quantization_width );
            for ( size_t vec_idx = 0; vec_idx < repetitions * K; vec_idx++ ) {
                std::vector<float> dir = sample_random_normal_vector( dimensions );
                random_vectors.push_back( dir.begin(), dir.end() );
                offsets.push_back( uniform( rng ) );
            }
        }

        static constexpr size_t get_concatenations() {
            return K;
        }

        size_t get_repetitions() const {
            return repetitions;
        }

        float get_quantization_width() const {
            return quantization_width;
        }

        void hash( typename Dataset::PointHandle point, std::vector<Value>& output ) const {
            output.resize( repetitions );
            const float inv_quantization_width = 1.0f / quantization_width;
            for ( size_t rep = 0; rep < repetitions; rep++ ) {
                BytewiseLshValue<K> cur;
                const size_t rep_base = K * rep;
                for ( size_t concat = 0; concat < K; concat++ ) {
                    const size_t idx = rep_base + concat;
                    typename Dataset::PointHandle rand_vec = random_vectors[idx];
                    float dotp = dot_product( point, rand_vec );
                    float quantized =
                        std::floor( ( dotp + offsets.at(idx) ) * inv_quantization_width );
                    int8_t code = static_cast<int8_t>( quantized );
                    cur.set( concat, code );
                }
                output.at(rep) = cur;
            }
        }

        float collision_probability( float distance ) const {
            distance = Distance::to_euclidean(distance); // This gives the chance of applying the square root
            float r = quantization_width;
            return 1.0 - 2.0 * normal_cdf( -r / distance ) -
                   ( 2.0 / ( std::sqrt( M_PI * 2.0 ) * ( r / distance ) ) ) *
                       ( 1.0 - std::exp( -r * r / ( 2.0 * distance * distance ) ) );
        }
    };

    //! A companion to `E2LSH`, sampling the hash of a uniformly random point
    //! of the ball `B(query, radius)` without ever materialising that point.
    //!
    //! Writing `A` for the matrix stacking the `K` projection vectors of one
    //! repetition, the point `u = q + r g / ||g||` with `g ~ N(0, I_d)` hashes
    //! to `floor( (A q + r A g / ||g|| + b) / w )`. Since `A g ~ N(0, A A^T)`,
    //! we can draw the projected displacement directly as `L z`, where `L` is
    //! the Cholesky factor of the Gram matrix `G = A A^T` and `z ~ N(0, I_K)`.
    //! Decomposing `g` into the part lying in the span of the rows of `A` and
    //! the part orthogonal to it gives `||g||^2 = ||z||^2 + s`, with
    //! `s ~ chi^2(d - K)`.
    //!
    //! Each sample therefore costs `K + 2` scalar draws and a `K x K`
    //! triangular product, instead of `d` draws and a `K x d` product.
    //!
    //! The sampler borrows the hasher it is built from, which must outlive it.
    template <uint8_t K, typename Dataset, typename Distance>
    class E2LSHBallSampler {
    public:
        using Hasher = E2LSH<K, Dataset, Distance>;
        using Value = typename Hasher::Value;
        using PointHandle = typename Dataset::PointHandle;

        //! The part of the hash that only depends on the query, namely `A q`.
        //! The projections are laid out like the projection vectors of the
        //! hasher, with `K` consecutive entries per repetition.
        struct PreparedQuery {
            std::vector<float> projections;
        };

    private:
        const Hasher* hasher = nullptr;
        size_t dimensions = 0;
        //! One `K x K` Cholesky factor per repetition.
        std::vector<CholeskyFactor> factors;

    public:
        E2LSHBallSampler( const Hasher& hasher ):
            hasher( &hasher ), dimensions( hasher.random_vectors.get_dimensions() ) {
            // The orthogonal component of `g` has `d - K` degrees of freedom.
            if ( dimensions <= K ) {
                throw std::invalid_argument(
                    "the number of concatenations must be smaller than the dimensionality" );
            }

            // The Gram matrix is built from the stored projection vectors
            // rather than from the floats they were created with, so that it
            // describes exactly the vectors `E2LSH::hash` uses, including any
            // normalization and quantization the dataset applies to them.
            std::vector<float> gram( static_cast<size_t>( K ) * K );
            factors.reserve( hasher.repetitions );
            for ( size_t rep = 0; rep < hasher.repetitions; rep++ ) {
                const size_t rep_base = K * rep;
                for ( size_t i = 0; i < K; i++ ) {
                    PointHandle row_i = hasher.random_vectors[rep_base + i];
                    for ( size_t j = 0; j <= i; j++ ) {
                        PointHandle row_j = hasher.random_vectors[rep_base + j];
                        float g = dot_product( row_i, row_j );
                        gram.at( i * K + j ) = g;
                        gram.at( j * K + i ) = g;
                    }
                }
                factors.emplace_back( gram, K );
            }
        }

        size_t get_repetitions() const {
            return hasher->get_repetitions();
        }

        //! Compute the query-dependent part of the hash, `A q`. This costs
        //! `K d` work per repetition, and can be reused across any number of
        //! samples drawn around the same query.
        PreparedQuery prepare( PointHandle query ) const {
            PreparedQuery prepared;
            prepared.projections.resize( hasher->offsets.size() );
            for ( size_t idx = 0; idx < prepared.projections.size(); idx++ ) {
                prepared.projections.at( idx ) =
                    dot_product( query, hasher->random_vectors[idx] );
            }
            return prepared;
        }

        //! Sample the hash, in repetition `rep`, of a uniformly random point of
        //! the ball of the given radius centered on the prepared query.
        Value sample_hash( const PreparedQuery& query,
                           float radius,
                           size_t rep,
                           std::mt19937_64& rng ) const {
            expect( rep < factors.size() );
            expect( query.projections.size() == hasher->offsets.size() );

            // The displacement in the projected space, `L z`, together with the
            // squared norm of the component of `g` lying in the span of `A`.
            std::array<float, K> z;
            float z_squared_norm = 0.0f;
            for ( size_t i = 0; i < K; i++ ) {
                float sample = sample_random_normal( rng );
                z[i] = sample;
                z_squared_norm += sample * sample;
            }
            std::array<float, K> displacement;
            factors.at( rep ).multiply( z.data(), displacement.data() );

            // The squared norm of the component of `g` orthogonal to the span
            // of `A`, which is a sum of `d - K` squared standard normals.
            std::chi_squared_distribution<float> chi_squared(
                static_cast<float>( dimensions - K ) );
            float orthogonal_squared_norm = chi_squared( rng );

            // The radius of the sampled point, distributed so that the point is
            // uniform in the ball rather than biased towards its center.
            float sampled_radius =
                radius * std::pow( sample_random_01( rng ), 1.0f / dimensions );
            float scale =
                sampled_radius / std::sqrt( z_squared_norm + orthogonal_squared_norm );

            const float inv_quantization_width = 1.0f / hasher->quantization_width;
            const size_t rep_base = K * rep;
            Value out;
            for ( size_t concat = 0; concat < K; concat++ ) {
                const size_t idx = rep_base + concat;
                float dotp = query.projections.at( idx ) + scale * displacement[concat];
                float quantized =
                    std::floor( ( dotp + hasher->offsets.at( idx ) ) * inv_quantization_width );
                out.set( concat, static_cast<int8_t>( quantized ) );
            }
            return out;
        }

        Value sample_hash( const PreparedQuery& query, float radius, size_t rep ) const {
            return sample_hash( query, radius, rep, get_global_rng() );
        }

        //! Convenience overload preparing the query on the fly. It pays `K d`
        //! work on every call, so prefer `prepare` when drawing more than a
        //! handful of samples around the same query.
        Value sample_hash( PointHandle query, float radius, size_t rep = 0 ) const {
            return sample_hash( prepare( query ), radius, rep );
        }
    };

    template <uint8_t K, typename Dataset, typename Distance>
    class E2LSHBuilder {
        float quantization_width = 0.0;
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
        using Output = E2LSH<K, Dataset, Distance>;

        //! `fit` rescales the hash function to a target distance
        static constexpr bool fits_to_distance = true;

        E2LSHBuilder() {
        }

        E2LSHBuilder( size_t dimensions ): quantization_width( 0 ), dimensions( dimensions ) {
        }

        E2LSHBuilder( float quantization_width, size_t dimensions ):
            quantization_width( quantization_width ), dimensions( dimensions ) {
        }

        void reset() {
            quantization_width = 0.0;
        }

        void fit(const Dataset& points, float dist, size_t reps, float delta) {
            fit(points, reps, delta);
        }

        void fit(const Dataset& points, size_t, float) {
            if (quantization_width != 0.0) {
                return;
            }

            Dataset fit_points = sample_points( points );
            if ( fit_points.size() == 0 ) {
                throw std::invalid_argument( "cannot fit hash builder on an empty dataset" );
            }

            Dataset random(dimensions);
            for (size_t i = 0; i < 1000; i++) {
                std::vector<float> dir = sample_random_normal_vector(dimensions);
                random.push_back(dir.begin(), dir.end());
            }

            float min = std::numeric_limits<float>::infinity();
            float max = -std::numeric_limits<float>::infinity();
            for (size_t i = 0; i < random.size(); i++) {
                for (size_t j = 0; j < fit_points.size(); j++) {
                    float dotp = dot_product(random[i], fit_points[j]);
                    if (dotp < min) min = dotp;
                    if (dotp > max) max = dotp;
                }
            }

            quantization_width = (max - min) / 16.0f;
            LOG_INFO("msg", "Quantization width guess", "quantization_width", quantization_width);

            const size_t sample_repetitions = 4;
            const size_t fit_n = fit_points.size();
            size_t high_thresh = static_cast<size_t>(sqrt(fit_n) * 1.3);
            size_t low_thresh  = static_cast<size_t>(sqrt(fit_n) * 0.7);

            std::optional<float> qw_lower = std::nullopt;
            std::optional<float> qw_upper = std::nullopt;

            auto compute_avg_collisions = [&](float qwidth) -> float {
                std::vector<PrefixMap<typename Output::Value>> pmaps(sample_repetitions);
                Output hasher(qwidth, dimensions, sample_repetitions);
                PrefixMap<typename Output::Value>::populate_from(pmaps, fit_points, hasher);

                size_t collisions = 0;
                for (auto& pmap : pmaps) {
                    PairPrefixMapCursorNew<typename Output::Value> cursor =
                        pmap.create_pair_cursor_new(hasher.get_concatenations(), std::nullopt);
                    collisions += cursor.total_collisions();
                }
                return static_cast<float>(collisions) / pmaps.size();
            };

            // Step 1: exponential search until both bounds known, starting from initial guess
            for (int iter = 0; iter < 10 && !(qw_lower && qw_upper); ++iter) {
                float avg_collisions = compute_avg_collisions(quantization_width);
                LOG_INFO("msg", "Exponential search quantization width",
                         "quantization_width", quantization_width,
                         "avg_collisions", avg_collisions,
                         "low_thresh", low_thresh,
                         "high_thresh", high_thresh);
                if (avg_collisions < low_thresh) {
                    qw_lower = quantization_width;           // too few collisions → increase width
                    quantization_width *= 2.0f;
                } else if (avg_collisions > high_thresh) {
                    qw_upper = quantization_width;           // too many collisions → decrease width
                    quantization_width /= 2.0f;
                } else {
                    // already in band
                    break;
                }
            }

            // Step 2: binary search between bounds
            if (qw_lower && qw_upper) {
                for (int iter = 0; iter < 5; ++iter) {
                    quantization_width = (*qw_lower + *qw_upper) / 2.0f;
                    float avg_collisions = compute_avg_collisions(quantization_width);
                    LOG_INFO("msg", "Binary search quantization width",
                             "quantization_width", quantization_width,
                             "avg_collisions", avg_collisions,
                             "low_thresh", low_thresh,
                             "high_thresh", high_thresh);

                    if (avg_collisions < low_thresh) {
                        *qw_lower = quantization_width;
                    } else if (avg_collisions > high_thresh) {
                        *qw_upper = quantization_width;
                    } else {
                        break; // within acceptable band
                    }

                    if (fabs(*qw_upper - *qw_lower) < 1e-7f) {
                        break; // converged
                    }
                }
            }

            // If the target is not met, just return to the intial guess
            // if (!qw_lower || !qw_upper) {
            //     quantization_width = (max - min) / 16.0f;
            // }

            LOG_INFO("msg", "Quantization width set to", "quantization_width", quantization_width);
        }

        void fit( Dataset& points, std::function<uint32_t( uint32_t )> group_fun ) {
            const float old_quantization_width = quantization_width;
            quantization_width = 0.0;
            std::vector<size_t> sampled_original_indices;
            Dataset fit_points = sample_points( points, &sampled_original_indices );
            if ( fit_points.size() == 0 ) {
                throw std::invalid_argument( "cannot fit hash builder on an empty dataset" );
            }

            const size_t fit_n = fit_points.size();
            const float diameter = approximate_diameter<Distance>( fit_points );
            const size_t sample_repetitions = 4;
            LOG_INFO( "diameter", diameter );

            std::vector<uint32_t> sampled_groups( fit_n );
            for ( size_t i = 0; i < fit_n; i++ ) {
                sampled_groups.at(i) = group_fun( sampled_original_indices.at(i) );
            }

            auto compute_avg_collisions = [&]( float qwidth ) -> float {
                std::vector<PrefixMap<typename Output::Value>> pmaps( sample_repetitions );
                Output hasher(qwidth, dimensions, sample_repetitions);
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
            const float threshold_low = std::sqrt(fit_n) / 2.0;
            const float threshold_high = fit_n * 10.0;
            LOG_INFO( "threshold-low", threshold_low, "threshold_high", threshold_high );

            float low = 2 * old_quantization_width;
            // Handle cold-start/corrupted-state runs where the previous width is zero.
            if ( low <= 0.0f ) {
                low = std::max( diameter / 16.0f, std::numeric_limits<float>::epsilon() );
            }
            float high = std::max( diameter, low * 1.01f );
            expect( low <= high );
            const size_t MAX_ITER = 40;
            bool found = false;
            for ( size_t iter = 0; iter < MAX_ITER; iter++ ) {
                float qwidth = ( low + high ) / 2.0;
                float avg_collisions = compute_avg_collisions( qwidth );
                LOG_INFO( "qwidth", qwidth, "avg-collisions", avg_collisions );
                if ( threshold_low <= avg_collisions && avg_collisions <= threshold_high ) {
                    quantization_width = qwidth;
                    found = true;
                    break;
                } else if ( avg_collisions < threshold_low ) {
                    low = qwidth;
                } else {
                    high = qwidth;
                }
            }
            if (!found) {
                quantization_width = std::max( low, std::numeric_limits<float>::epsilon() );
            }
            LOG_INFO( "quantization-width", quantization_width );
            expect( quantization_width > 0.0f );
        }


        Output build( size_t repetitions ) const {
            expect( quantization_width > 0 );
            return E2LSH<K, Dataset, Distance>( quantization_width, dimensions, repetitions );
        }

        std::string describe() const {
            std::stringstream sstream;
            sstream << "E2LSH(quantization=" << quantization_width << ")";
            return sstream.str();
        }
    };

} // namespace panna
