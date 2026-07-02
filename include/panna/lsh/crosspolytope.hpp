#pragma once
#include <algorithm>
#include <cstdint>
#include <omp.h>
#include <random>
#include <sstream>
#include <vector>

#include "panna/linalg.hpp"
#include "panna/lsh/values.hpp"

namespace panna {
    struct CrossPolytopeCollisionEstimates {
        std::vector<float> probabilities;
        float eps;

        CrossPolytopeCollisionEstimates() {
        }

        template <typename Archive>
        void serialize( Archive& ar ) {
            ar( probabilities, eps );
        }

        friend bool operator==( const CrossPolytopeCollisionEstimates& a,
                                const CrossPolytopeCollisionEstimates& b ) {
            return a.probabilities == b.probabilities && a.eps == b.eps;
        }

        CrossPolytopeCollisionEstimates( unsigned int dimensions,
                                         unsigned int num_repetitions,
                                         float eps ):
            eps( eps ) {
            // adapted from
            // https://github.com/puffinn/puffinn/blob/master/include/puffinn/hash/crosspolytope.hpp

            auto log_dimensions = ceil_log( dimensions );
            probabilities = std::vector<float>();
            size_t num_probabilities = 2 / eps;
            probabilities.resize( num_probabilities );

            // foreach [alpha, alpha+eps) segment
#pragma omp parallel for
            for ( size_t i = 0; i < num_probabilities; i++ ) {
                float alpha = -1 + eps * i;
                std::normal_distribution<double> standard_normal( 0, 1 );
                std::mt19937_64 rng;
                rng.seed( 1234 + i );

                size_t collisions = 0;

                for ( size_t z = 0; z < num_repetitions; z++ ) {
                    // length = dimensions
                    // x = (1, 0, ..., 0)
                    // y = (alpha, (1-alpha^2)^(1/2), ..., 0)
                    // The dot product between x and y is alpha, and both are
                    // unit norm.

                    // The hash value so far.
                    uint32_t hash_x = 0;
                    uint32_t hash_y = 0;
                    // Absolute value of highest value seen.
                    double v_x = 0;
                    double v_y = 0;

                    // Compute a random rotation of x and y using the matrix z
                    // [ [ z_1_0, z_2_0 ],
                    //   [ z_1_1, z_2_1 ],
                    //   [ z_1_j, z_2_j ] ]
                    for ( uint32_t j = 0; j < dimensions; j++ ) {
                        double z_1 = standard_normal( rng );
                        double z_2 = standard_normal( rng );
                        // calculate z*x[j] and find the index with the highest
                        // value
                        if ( abs( z_1 ) > v_x ) {
                            v_x = abs( z_1 );
                            hash_x = j;
                            if ( z_1 < 0 ) {
                                hash_x |= ( 1 << log_dimensions );
                            }
                        }
                        // do the same for z*y[j]
                        double h_y = alpha * z_1 + pow( 1 - pow( alpha, 2 ), 0.5 ) * z_2;
                        if ( abs( h_y ) > v_y ) {
                            v_y = abs( h_y );
                            hash_y = j;
                            if ( h_y < 0 ) {
                                hash_y |= ( 1 << log_dimensions );
                            }
                        }
                    }
                    collisions += hash_x == hash_y;
                }
                float prob;
                if ( num_repetitions != 0 ) {
                    prob = static_cast<float>( collisions ) / num_repetitions;
                } else {
                    prob = 1.0;
                }
                assert( prob <= 1.0 );
                probabilities.at(i) = prob;
            }
        }

        float get_collision_probability( float dotp ) const {
            size_t idx = std::floor( ( 1.0 + dotp ) / eps );
            return probabilities.at(idx);
        }
    };

    template <uint8_t K, typename Dataset, typename Distance>
    class CrossPolytopeBuilder;

    template <uint8_t K, typename Dataset, typename Distance>
    class CrossPolytope {
    public:
        //! The datatype of the output
        using Value = ShortLshValue<K>;
        using Builder = CrossPolytopeBuilder<K, Dataset, Distance>;

    private:
        size_t repetitions;
        size_t dimensions;
        std::vector<RandomDotProducts> random_dots;
        // scratch space
        std::vector<std::vector<float>> tl_rotated_vectors;
        CrossPolytopeCollisionEstimates estimates;

        int16_t encode_closest_axis( const std::vector<float>& vec ) const {
            int res = 0;
            float max_sim = 0;
            for ( size_t i = 0; i < dimensions; i++ ) {
                if ( vec.at(i) > max_sim ) {
                    res = i;
                    max_sim = vec.at(i);
                } else if ( -vec.at(i) > max_sim ) {
                    // TODO: check if -i works as well
                    res = i + dimensions;
                    max_sim = -vec.at(i);
                }
            }
            return res;
        }

        // Hash a single repetition of the given vector
        int16_t hash_single( std::vector<float>& vec, size_t concatenation, size_t repetition ) {
            size_t idx = repetition * K + concatenation;
            random_dots.at(idx).compute( vec );
            return encode_closest_axis( vec );
        }

    public:
        static constexpr size_t get_concatenations() {
            return K;
        }

        CrossPolytope() {
        }
        CrossPolytope( size_t dimensions,
                       size_t repetitions,
                       size_t estimation_repetitions = 2 * 4096,
                       float estimation_eps = 5e-3 ):
            repetitions( repetitions ),
            dimensions( dimensions ),
            estimates( ( 1 << ceil_log( dimensions ) ), estimation_repetitions, estimation_eps ) {

            for ( size_t i = 0; i < K * repetitions; i++ ) {
                random_dots.emplace_back( dimensions );
            }

            // prepare thread local scratch space
            for ( int i = 0; i < omp_get_max_threads(); i++ ) {
                tl_rotated_vectors.push_back( random_dots.at(0).allocate_scratch() );
            }
        }

        template <typename Archive>
        void serialize( Archive& ar ) {
            ar( repetitions, dimensions, random_dots, tl_rotated_vectors, estimates );
        }

        friend bool operator==( const CrossPolytope<K, Dataset, Distance>& a,
                                const CrossPolytope<K, Dataset, Distance>& b ) {
            return a.repetitions == b.repetitions && a.dimensions == b.dimensions &&
                   a.random_dots == b.random_dots && a.estimates == b.estimates;
        }

        //! Requires Dataset::PointHandle to support the operation
        //! `PointHandle::into_vec` that copies the contents of the
        //! point into the given vector, without changing its length.
        void hash( typename Dataset::PointHandle point, std::vector<Value>& output ) {
            auto& rotated_vector = tl_rotated_vectors.at(omp_get_thread_num());
            output.clear();
            Value cur;
            for ( size_t rep = 0; rep < repetitions; rep++ ) {
                for ( size_t concat = 0; concat < K; concat++ ) {
                    point.into_vec( rotated_vector );
                    std::fill( rotated_vector.begin() + dimensions, rotated_vector.end(), 0.0 );
                    int16_t code = hash_single( rotated_vector, concat, rep );
                    cur.set( concat, code );
                }
                output.push_back( cur );
                cur = Value();
            }
        }

        float collision_probability( float distance ) const {
            float dotp = Distance::to_dot_product( distance );
            return estimates.get_collision_probability( dotp );
        }
    };

    template <uint8_t K, typename Dataset, typename Distance>
    class CrossPolytopeBuilder {
        size_t dimensions = 0;
        size_t estimation_repetitions = 1024;
        float estimation_eps = 5e-3;

    public:
        using Output = CrossPolytope<K, Dataset, Distance>;

        CrossPolytopeBuilder() {
        }

        CrossPolytopeBuilder( size_t dimensions,
                              size_t estimation_repetitions = 1024,
                              float estimation_eps = 5e-3 ):
            dimensions( dimensions ),
            estimation_repetitions( estimation_repetitions ),
            estimation_eps( estimation_eps ) {
        }

        template <typename Archive>
        void serialize( Archive& ar ) {
            ar(dimensions, estimation_repetitions, estimation_eps);
        }

        void fit( Dataset&, size_t, float ) {
        }

        Output build( size_t repetitions ) const {
            return CrossPolytope<K, Dataset, Distance>(
                dimensions, repetitions, estimation_repetitions, estimation_eps );
        }

        std::string describe() const {
            std::stringstream sstream;
            sstream << "CrossPolytope";
            return sstream.str();
        }
    };

} // namespace panna
