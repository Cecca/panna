#pragma once

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cmath>
#include <unordered_map>
#include <unordered_set>

#include "panna/data.hpp"
#include "panna/distance.hpp"
#include "panna/lsh/crosspolytope.hpp"
#include "panna/lsh/euclidean.hpp"
#include "panna/lsh/minhash.hpp"
#include "panna/lsh/predicates.hpp"
#include "panna/lsh/simhash.hpp"
#include "panna/lsh/tensoring.hpp"
#include "panna/rand.hpp"

namespace panna {

    template <typename Dataset, typename Distance, typename HasherBuilder>
    static void test_hash_collision_probability( HasherBuilder builder,
                                                 unsigned int dimensions,
                                                 unsigned int repetitions,
                                                 unsigned int num_experiments = 1000,
                                                 float accepted_deviation = 0.05 ) {
        using Hasher = typename HasherBuilder::Output;
        seed_global_rng( 1234 );

        Hasher hasher = builder.build( repetitions );

        float total_expected = 0;
        float total_actual = 0;
        std::vector<typename Hasher::Value> output_a;
        std::vector<typename Hasher::Value> output_b;
        for ( unsigned int i = 0; i < num_experiments; i++ ) {
            Dataset dataset( dimensions );
            dataset.push_back_random();
            dataset.push_back_random();

            typename Dataset::PointHandle vec_a = dataset[0];
            typename Dataset::PointHandle vec_b = dataset[1];

            hasher.hash( vec_a, output_a );
            hasher.hash( vec_b, output_b );
            float dist = Distance::compute( vec_a, vec_b );
            float prob = hasher.collision_probability( dist );
            float empirical = 0.0;
            REQUIRE( output_a.size() == repetitions );
            REQUIRE( output_b.size() == repetitions );
            for ( size_t i = 0; i < repetitions; i++ ) {
                if ( output_a[i] == output_b[i] ) {
                    empirical += 1;
                }
            }
            empirical /= repetitions;
            total_actual += empirical;
            total_expected += prob;
        }
        REQUIRE(std::abs(total_actual - total_expected) / num_experiments <= accepted_deviation);
    }

    TEST_CASE( "Simhash collision probability" ) {
        using Dataset = UnitNormPoints;
        for ( size_t dimensions : { 10, 100, 200 } ) {
            SimhashBuilder<1, Dataset, CosineDistance> builder( dimensions );
            test_hash_collision_probability<Dataset,
                                            CosineDistance,
                                            SimhashBuilder<1, Dataset, CosineDistance>>(
                builder, dimensions, 4096 );
        }
    }

    TEST_CASE( "Simhash collision probability (angular distance)" ) {
        using Dataset = UnitNormPoints;
        using Distance = AngularDistance;
        for ( size_t dimensions : { 10, 100, 200 } ) {
            SimhashBuilder<1, Dataset, Distance> builder( dimensions );
            test_hash_collision_probability<Dataset,
                                            Distance,
                                            SimhashBuilder<1, Dataset, Distance>>(
                builder, dimensions, 4096 );
        }
    }

    TEST_CASE( "CrossPolytope collision probability" ) {
        using Dataset = UnitNormPoints;
        using Distance = CosineDistance;
        for ( size_t dimensions : { 10, 100, 200 } ) {
            CrossPolytopeBuilder<1, Dataset, Distance> builder( dimensions, 8192 );
            test_hash_collision_probability<Dataset,
                                            Distance,
                                            CrossPolytopeBuilder<1, Dataset, Distance>>(
                builder, dimensions, 4096 );
        }
    }

    TEST_CASE( "E2LSH collision probability" ) {
        using Dataset = EuclideanPoints;
        float r = 1.0;
        for ( size_t dimensions : { 10, 100, 200 } ) {
            E2LSHBuilder<1, Dataset, EuclideanDistance> builder( r, dimensions );
            test_hash_collision_probability<Dataset, EuclideanDistance, E2LSHBuilder<1, Dataset, EuclideanDistance>>(
                builder, dimensions, 4096 );
        }
    }

    //! Draw a point uniformly at random from the ball of the given radius
    //! centered on `query`, and hash it the straightforward way.
    template <uint8_t K>
    static BytewiseLshValue<K>
    hash_explicit_ball_sample( const E2LSH<K, EuclideanPoints, EuclideanDistance>& hasher,
                               const std::vector<float>& query,
                               float radius,
                               size_t rep,
                               EuclideanPoints& scratch,
                               std::vector<BytewiseLshValue<K>>& output ) {
        const size_t dimensions = query.size();
        std::vector<float> direction = sample_random_normal_vector( dimensions );
        normalize( direction );
        float sampled_radius =
            radius * std::pow( sample_random_01(), 1.0f / dimensions );

        std::vector<float> point( dimensions );
        for ( size_t i = 0; i < dimensions; i++ ) {
            point.at( i ) = query.at( i ) + sampled_radius * direction.at( i );
        }

        scratch.clear();
        scratch.push_back( point.begin(), point.end() );
        hasher.hash( scratch[0], output );
        return output.at( rep );
    }

    //! Total variation distance between two histograms over hash values.
    template <typename Value>
    static float total_variation( const std::unordered_map<Value, size_t>& a,
                                  const std::unordered_map<Value, size_t>& b,
                                  size_t samples ) {
        std::unordered_set<Value> keys;
        for ( const auto& entry : a ) {
            keys.insert( entry.first );
        }
        for ( const auto& entry : b ) {
            keys.insert( entry.first );
        }

        float distance = 0.0;
        for ( const Value& key : keys ) {
            auto in_a = a.find( key );
            auto in_b = b.find( key );
            float pa = ( in_a == a.end() ) ? 0.0f : static_cast<float>( in_a->second ) / samples;
            float pb = ( in_b == b.end() ) ? 0.0f : static_cast<float>( in_b->second ) / samples;
            distance += std::abs( pa - pb );
        }
        return distance / 2.0;
    }

    //! With a zero radius the sampled point is the query itself, so the sampled
    //! hash must reproduce the plain hash exactly. This pins down the
    //! bookkeeping of the `A q + b` terms, independently of the random part.
    template <typename Dataset>
    static void test_ball_sampler_zero_radius() {
        constexpr uint8_t K = 4;
        const size_t dimensions = 64;
        const size_t repetitions = 8;
        seed_global_rng( 1234 );

        E2LSH<K, Dataset, EuclideanDistance> hasher( 1.5, dimensions, repetitions );
        E2LSHBallSampler<K, Dataset, EuclideanDistance> sampler( hasher );
        REQUIRE( sampler.get_repetitions() == repetitions );

        Dataset queries( dimensions );
        queries.push_back_random();

        std::vector<BytewiseLshValue<K>> expected;
        hasher.hash( queries[0], expected );

        auto prepared = sampler.prepare( queries[0] );
        for ( size_t rep = 0; rep < repetitions; rep++ ) {
            REQUIRE( sampler.sample_hash( prepared, 0.0, rep ) == expected.at( rep ) );
            REQUIRE( sampler.sample_hash( queries[0], 0.0, rep ) == expected.at( rep ) );
        }
    }

    TEST_CASE( "E2LSHBallSampler with zero radius reproduces the hash" ) {
        // Both dataset layouts E2LSH is instantiated with, since each stores
        // the projection vectors differently and hence yields a different Gram
        // matrix.
        test_ball_sampler_zero_radius<EuclideanPoints>();
        test_ball_sampler_zero_radius<NormedPoints>();
    }

    //! The hashes sampled directly in the projected space must follow the same
    //! distribution as the hashes of points explicitly sampled in the ball.
    template <uint8_t K>
    static void test_ball_sampler_distribution( size_t dimensions,
                                                float radius,
                                                float quantization_width,
                                                float accepted_deviation = 0.03 ) {
        using Dataset = EuclideanPoints;
        using Value = BytewiseLshValue<K>;
        const size_t samples = 200000;
        const size_t rep = 0;
        seed_global_rng( 1234 );

        E2LSH<K, Dataset, EuclideanDistance> hasher( quantization_width, dimensions, 1 );
        E2LSHBallSampler<K, Dataset, EuclideanDistance> sampler( hasher );

        Dataset queries( dimensions );
        queries.push_back_random();
        std::vector<float> query( dimensions );
        queries[0].into_vec( query );

        auto prepared = sampler.prepare( queries[0] );
        std::unordered_map<Value, size_t> sampled;
        for ( size_t i = 0; i < samples; i++ ) {
            sampled[sampler.sample_hash( prepared, radius, rep )] += 1;
        }

        std::unordered_map<Value, size_t> explicitly_sampled;
        Dataset scratch( dimensions );
        std::vector<Value> output;
        for ( size_t i = 0; i < samples; i++ ) {
            explicitly_sampled[hash_explicit_ball_sample<K>(
                hasher, query, radius, rep, scratch, output )] += 1;
        }

        // A degenerate run, where everything lands in one bucket, would pass
        // the comparison below without testing anything.
        REQUIRE( sampled.size() > 4 );

        float distance = total_variation( sampled, explicitly_sampled, samples );
        LOG_INFO( "concatenations", static_cast<int>( K ),
                  "buckets", sampled.size(),
                  "total-variation", distance );
        REQUIRE( distance <= accepted_deviation );
    }

    //! A single projection, checking the marginal distribution and in
    //! particular the interplay of the chi-squared term with the sampled
    //! radius.
    TEST_CASE( "E2LSHBallSampler marginal distribution" ) {
        test_ball_sampler_distribution<1>( 64, 2.0, 1.0 );
    }

    //! The joint distribution of several projections, which is what the
    //! Cholesky factor is there for. The dimensionality is deliberately small:
    //! with many dimensions the projection vectors are nearly orthogonal, the
    //! Gram matrix is nearly diagonal, and the test loses the power to tell a
    //! correct factor from, say, its transpose.
    TEST_CASE( "E2LSHBallSampler joint distribution" ) {
        test_ball_sampler_distribution<4>( 8, 2.0, 2.0 );
    }

    TEST_CASE( "MinHash collision probability" ) {
        using Dataset = SparseSets;
        for ( size_t dimensions : { 20000 } ) {
            MinhashBuilder<1, Dataset> builder;
            test_hash_collision_probability<Dataset, JaccardDistance, MinhashBuilder<1, Dataset>>(
                builder, dimensions, 4096 );
        }
    }

    TEST_CASE( "MinHash1Bit collision probability" ) {
        using Dataset = SparseSets;
        for ( size_t dimensions : { 200000 } ) {
            Minhash1BitBuilder<1, Dataset> builder;
            test_hash_collision_probability<Dataset,
                                            JaccardDistance,
                                            Minhash1BitBuilder<1, Dataset>>(
                builder, dimensions, 4096, 10 );
        }
    }

    template <typename Dataset, typename Distance, typename HasherBuilder>
    static void test_failure_probability( HasherBuilder builder,
                                          unsigned int dimensions,
                                          unsigned int repetitions,
                                          unsigned int num_experiments = 1,
                                          size_t samples_per_experiment = 1000 ) {

        using Hasher = typename HasherBuilder::Output;
        seed_global_rng( 1234 );

        Hasher hasher = builder.build( repetitions );
        size_t max_concats = hasher.get_concatenations();

        std::vector<typename Hasher::Value> output_a;
        std::vector<typename Hasher::Value> output_b;
        for ( unsigned int experiment = 0; experiment < num_experiments; experiment++ ) {
            Dataset dataset( dimensions );
            dataset.push_back_random();
            dataset.push_back_random();
            typename Dataset::PointHandle vec_a = dataset[0];
            typename Dataset::PointHandle vec_b = dataset[1];
            float dist = Distance::compute( vec_a, vec_b );

            std::vector<std::vector<float>> counters;
            for ( size_t k = 1; k <= max_concats; k++ ) {
                std::vector<float> counters_row( repetitions );
                std::fill( counters_row.begin(), counters_row.end(), 0.0 );
                counters.push_back( counters_row );
            }

            std::vector<std::vector<float>> expected;
            for ( size_t k = 1; k <= max_concats; k++ ) {
                std::vector<float> expected_row;
                for ( size_t rep = 1; rep <= repetitions; rep++ ) {
                    float fp =
                        failure_probability( hasher, dist, k, rep, hasher.get_repetitions() );
                    expected_row.push_back( fp );
                }
                expected.push_back( expected_row );
            }

            for ( size_t sample = 0; sample < samples_per_experiment; sample++ ) {
                hasher = builder.build( repetitions );
                hasher.hash( vec_a, output_a );
                hasher.hash( vec_b, output_b );
                for ( size_t k = 1; k <= max_concats; k++ ) {
                    for ( size_t rep = 0; rep < repetitions; rep++ ) {
                        bool found = false;
                        for ( size_t i = 0; i < rep; i++ ) {
                            found |= output_a[i].prefix_eq( output_b[i], k );
                            if ( found ) {
                                break;
                            }
                        }
                        if ( k < max_concats && !found ) {
                            for ( size_t i = rep; i < repetitions; i++ ) {
                                found |= output_a[i].prefix_eq( output_b[i], k + 1 );
                                if ( found ) {
                                    break;
                                }
                            }
                        }
                        if ( !found ) {
                            counters[k - 1][rep] += 1;
                        }
                    }
                }
            }

            for ( size_t k = 1; k <= max_concats; k++ ) {
                for ( size_t rep = 0; rep < repetitions; rep++ ) {
                    float actual = counters[k - 1][rep] / samples_per_experiment;
                    float exp = expected[k - 1][rep];
                    // We check that the actual failure probability is lower than the expected one
                    REQUIRE( actual < exp + 0.03 );
                }
            }
        }
    }

    TEST_CASE( "Failure probability independent" ) {
        using Dataset = UnitNormPoints;
        for ( size_t dimensions : { 10 } ) {
            SimhashBuilder<24, Dataset, CosineDistance> builder( dimensions );
            test_failure_probability<Dataset,
                                     CosineDistance,
                                     SimhashBuilder<24, Dataset, CosineDistance>>(
                builder, dimensions, 128 );
        }
    }

    TEST_CASE( "Failure probability tensoring" ) {
        using Dataset = UnitNormPoints;
        for ( size_t dimensions : { 10 } ) {
            SimhashBuilder<12, Dataset, CosineDistance> inner_builder( dimensions );
            TensoringBuilder<SimhashBuilder<12, Dataset, CosineDistance>, Dataset> builder(
                inner_builder );
            test_failure_probability<
                Dataset,
                CosineDistance,
                TensoringBuilder<SimhashBuilder<12, Dataset, CosineDistance>, Dataset>>(
                builder, dimensions, 128 );
        }
    }

    TEST_CASE( "Tensoring" ) {
        using Dataset = UnitNormPoints;
        using Distance = CosineDistance;
        using Hasher = CrossPolytope<2, Dataset, Distance>;
        using TensoredHasher = Tensoring<Hasher, Dataset>;

        const uint8_t K_HALF = 2;
        const size_t dimensions = 128;
        CrossPolytopeBuilder<K_HALF, Dataset, Distance> builder( dimensions );

        TensoredHasher tensored( builder, 4096 );
        static_assert( tensored.get_concatenations() == 2 * K_HALF,
                       "the tensored dataset should have twice the concatenations" );

        Dataset dataset( dimensions );
        dataset.push_back_random();
        std::vector<TensoredHasher::Value> hash_a;
        std::vector<TensoredHasher::Value> hash_b;
        tensored.hash( dataset[0], hash_a );
        tensored.hash( dataset[0], hash_b );

        REQUIRE( hash_a == hash_b );

        // for ( float d : { 0.1, 0.2, 0.3 } ) {
        //     float fp = failure_probability( tensored, d, 4, 1, tensored.get_repetitions() );
        // }
    }

} // namespace panna
