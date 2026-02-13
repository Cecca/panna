#pragma once
#include <limits>
#include <optional>
#include <random>
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
    class E2LSH {
    public:
        //! The datatype of the output
        using Value = BytewiseLshValue<K>;
        using Builder = E2LSHBuilder<K, Dataset, Distance>;

    private:
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

        template <typename Archive>
        void serialize( Archive& ar ) {
            ar( quantization_width, repetitions, random_vectors, offsets );
        }

        static constexpr size_t get_concatenations() {
            return K;
        }

        size_t get_repetitions() const {
            return repetitions;
        }

        void hash( typename Dataset::PointHandle point, std::vector<Value>& output ) const {
            output.clear();
            BytewiseLshValue<K> cur;
            for ( size_t rep = 0; rep < repetitions; rep++ ) {
                for ( size_t concat = 0; concat < K; concat++ ) {
                    typename Dataset::PointHandle rand_vec = random_vectors[K * rep + concat];
                    float dotp = dot_product( point, rand_vec );
                    float quantized =
                        std::floor( ( dotp + offsets[K * rep + concat] ) / quantization_width );
                    int8_t code = static_cast<int8_t>( quantized );
                    cur.set( concat, code );
                }
                output.push_back( cur );
                cur = BytewiseLshValue<K>();
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

    template <uint8_t K, typename Dataset, typename Distance>
    class E2LSHBuilder {
        float quantization_width = 0.0;
        size_t dimensions = 0;

    public:
        using Output = E2LSH<K, Dataset, Distance>;

        E2LSHBuilder() {
        }

        E2LSHBuilder( size_t dimensions ): quantization_width( 0 ), dimensions( dimensions ) {
        }

        E2LSHBuilder( float quantization_width, size_t dimensions ):
            quantization_width( quantization_width ), dimensions( dimensions ) {
        }

        template <typename Archive>
        void serialize( Archive& ar ) {
            ar( quantization_width, dimensions );
        }

        void fit(Dataset& points) {
            if (quantization_width != 0.0) {
                return;
            }

            Dataset random(dimensions);
            for (size_t i = 0; i < 1000; i++) {
                std::vector<float> dir = sample_random_normal_vector(dimensions);
                random.push_back(dir.begin(), dir.end());
            }

            float min = std::numeric_limits<float>::infinity();
            float max = -std::numeric_limits<float>::infinity();
            for (size_t i = 0; i < random.size(); i++) {
                for (size_t j = 0; j < points.size(); j++) {
                    float dotp = dot_product(random[i], points[j]);
                    if (dotp < min) min = dotp;
                    if (dotp > max) max = dotp;
                }
            }

            quantization_width = (max - min) / 16.0f;
            LOG_INFO("msg", "Quantization width guess", "quantization_width", quantization_width);

            const size_t sample_repetitions = 4;
            size_t high_thresh = static_cast<size_t>(sqrt(points.size()) * 1.3);
            size_t low_thresh  = static_cast<size_t>(sqrt(points.size()) * 0.7);

            std::optional<float> qw_lower = std::nullopt;
            std::optional<float> qw_upper = std::nullopt;

            auto compute_avg_collisions = [&](float qwidth) -> float {
                std::vector<PrefixMap<typename Output::Value>> pmaps(sample_repetitions);
                Output hasher(qwidth, dimensions, sample_repetitions);
                PrefixMap<typename Output::Value>::populate_from(pmaps, points, hasher);

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
                for (int iter = 0; iter < 15; ++iter) {
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


        /// Widen the quantization width by the given factor, to increase
        /// the collision probability in a subsequent rehashing round.
        void widen( float factor ) {
            expect( quantization_width > 0 );
            quantization_width *= factor;
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
