#pragma once

//! Tests for `pair_forest_emst` (`panna/pairemst.hpp`), the OpenMP-driven EMST
//! search over `PairForestIndex`.
//!
//! The checks mirror the ones `emst_phase0_test.hpp` makes for the existing
//! search: a PIN against the exact tree at `epsilon = 0`, the `(1 + epsilon)`
//! invariant at `epsilon = 0.2`, and structural checks that the result really
//! is a spanning tree. Degenerate shapes -- duplicate points, fewer points than
//! a batch has repetitions, fewer repetitions than a batch -- get their own
//! case, because the batching is exactly where those would break.

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <omp.h>
#include <random>
#include <stdexcept>
#include <vector>

#include "panna/data.hpp"
#include "panna/distance.hpp"
#include "panna/dsu.hpp"
#include "panna/emst_common.hpp"
#include "panna/lsh/lattice.hpp"
#include "panna/pairemst.hpp"
#include "panna/rand.hpp"

namespace panna {
    namespace pairemst_test {

        using Dataset = EuclideanPoints;
        using Distance = EuclideanDistance;
        using Hasher = LatticeLSH<4, Dataset, Distance>;

        //! A random gaussian dataset with a fixed seed, so every case below is
        //! reproducible independently of the order Catch2 runs them in.
        inline Dataset make_dataset( size_t dimensions, size_t n, uint64_t seed ) {
            seed_global_rng( seed );
            Dataset data( dimensions );
            for ( size_t i = 0; i < n; i++ ) {
                data.push_back_random();
            }
            return data;
        }

        inline float tree_weight( const std::vector<Edge>& tree ) {
            float w = 0.0f;
            for ( const Edge& e : tree ) {
                w += e.weight;
            }
            return w;
        }

        //! Everything a spanning tree must be, whatever the parameters were:
        //! `n - 1` edges, ascending, no self-loop, and actually connected.
        inline void require_spanning_tree( const std::vector<Edge>& tree, size_t n ) {
            REQUIRE( tree.size() == n - 1 );
            REQUIRE( std::is_sorted( tree.begin(), tree.end() ) );

            DSU dsu( static_cast<uint32_t>( n ) );
            size_t self_loops = 0;
            size_t cycles = 0;
            for ( const Edge& e : tree ) {
                if ( e.a == e.b ) {
                    self_loops++;
                } else if ( !dsu.union_sets( e.a, e.b ) ) {
                    cycles++;
                }
            }
            REQUIRE( self_loops == 0 );
            REQUIRE( cycles == 0 );
            REQUIRE( dsu.num_connected_components() == 1 );
        }

    } // namespace pairemst_test

    // -----------------------------------------------------------------------
    // PIN — at epsilon = 0 the search may not stop before the tree is exact,
    // so its weight must equal the weight of the exact MST.
    // -----------------------------------------------------------------------
    TEST_CASE( "pair_forest_emst matches the exact tree at epsilon=0", "[pairemst]" ) {
        using namespace pairemst_test;

        const size_t dimensions = 8;
        const size_t n = 200;
        const size_t repetitions = 512;

        const Dataset data = make_dataset( dimensions, n, 1 );
        const auto exact = exact_emst<Dataset, Distance>( data );

        const PairEmstResult result =
            pair_forest_emst<Dataset, Hasher, Distance>( data, 0.0f, 0.001f, repetitions );

        require_spanning_tree( result.tree, n );
        REQUIRE( result.weight == exact.first );
        REQUIRE( tree_weight( result.tree ) == exact.first );
        REQUIRE( result.distances_computed > 0 );
        REQUIRE( result.repetitions_at_stop > 0 );
    }

    // -----------------------------------------------------------------------
    // INVARIANT — at epsilon = 0.2 the realised ratio must stay inside the
    // promised bound, and may never undershoot the true MST weight.
    // -----------------------------------------------------------------------
    TEST_CASE( "pair_forest_emst respects the (1+epsilon) bound", "[pairemst]" ) {
        using namespace pairemst_test;

        const size_t dimensions = 8;
        const size_t n = 200;
        const size_t repetitions = 512;
        const float epsilon = 0.2f;
        // Slack for float accumulation over a few hundred edges.
        const float tolerance = 1e-4f;

        for ( uint64_t seed : { 1, 2, 3 } ) {
            INFO( "seed=" << seed );
            const Dataset data = make_dataset( dimensions, n, seed );
            const float exact_weight = exact_emst<Dataset, Distance>( data ).first;
            REQUIRE( exact_weight > 0.0f );

            const PairEmstResult result =
                pair_forest_emst<Dataset, Hasher, Distance>( data, epsilon, 0.01f, repetitions );

            require_spanning_tree( result.tree, n );
            CHECK( result.weight >= exact_weight - tolerance );
            CHECK( result.weight <= ( 1.0f + epsilon ) * exact_weight + tolerance );
        }
    }

    // -----------------------------------------------------------------------
    // Degenerate shapes. Each of these stresses a different seam of the
    // batching, so they are asserted separately rather than in a loop.
    // -----------------------------------------------------------------------
    TEST_CASE( "pair_forest_emst handles degenerate inputs", "[pairemst]" ) {
        using namespace pairemst_test;

        SECTION( "duplicate points give zero-weight edges, not a broken tree" ) {
            // Ten distinct positions, each repeated five times: half the true
            // MST has weight exactly zero, which is the case a cutoff with a
            // multiplicative-only slack would get wrong.
            const size_t dimensions = 4;
            seed_global_rng( 77 );
            Dataset data( dimensions );
            std::vector<std::vector<float>> distinct;
            for ( size_t i = 0; i < 10; i++ ) {
                distinct.push_back( sample_random_normal_vector( dimensions ) );
            }
            for ( size_t copy = 0; copy < 5; copy++ ) {
                for ( const auto& p : distinct ) {
                    data.push_back( p.begin(), p.end() );
                }
            }
            const size_t n = data.size();

            const float exact_weight = exact_emst<Dataset, Distance>( data ).first;
            const PairEmstResult result =
                pair_forest_emst<Dataset, Hasher, Distance>( data, 0.0f, 0.01f, 256 );

            require_spanning_tree( result.tree, n );
            REQUIRE( result.tree.front().weight == 0.0f );
            REQUIRE( result.weight == exact_weight );
        }

        SECTION( "fewer points than a batch has repetitions" ) {
            const size_t n = PAIR_EMST_BATCH_REPETITIONS / 2;
            const Dataset data = make_dataset( 4, n, 555 );
            const float exact_weight = exact_emst<Dataset, Distance>( data ).first;

            const PairEmstResult result =
                pair_forest_emst<Dataset, Hasher, Distance>( data, 0.0f, 0.01f, 256 );

            require_spanning_tree( result.tree, n );
            REQUIRE( result.weight == exact_weight );
        }

        SECTION( "fewer repetitions than a batch is wide" ) {
            // The last (and only) batch is a partial one: `width < 32`.
            const size_t n = 120;
            const size_t repetitions = 12;
            REQUIRE( repetitions < PAIR_EMST_BATCH_REPETITIONS );

            const Dataset data = make_dataset( 4, n, 909 );
            const float exact_weight = exact_emst<Dataset, Distance>( data ).first;

            const PairEmstResult result =
                pair_forest_emst<Dataset, Hasher, Distance>( data, 0.2f, 0.05f, repetitions );

            require_spanning_tree( result.tree, n );
            CHECK( result.weight >= exact_weight - 1e-4f );
            CHECK( result.weight <= 1.2f * exact_weight + 1e-4f );
            REQUIRE( result.repetitions_at_stop <= repetitions );
        }

        SECTION( "a repetition count that is not a multiple of the batch width" ) {
            const size_t n = 120;
            const size_t repetitions = PAIR_EMST_BATCH_REPETITIONS * 2 + 5;
            const Dataset data = make_dataset( 4, n, 31337 );
            const float exact_weight = exact_emst<Dataset, Distance>( data ).first;

            const PairEmstResult result =
                pair_forest_emst<Dataset, Hasher, Distance>( data, 0.0f, 0.01f, repetitions );

            require_spanning_tree( result.tree, n );
            REQUIRE( result.weight == exact_weight );
        }

        SECTION( "fewer than two points is rejected" ) {
            const Dataset one = make_dataset( 4, 1, 4 );
            REQUIRE_THROWS_AS(
                ( pair_forest_emst<Dataset, Hasher, Distance>( one, 0.0f, 0.01f, 16 ) ),
                std::invalid_argument );
        }
    }

    TEST_CASE( "sort_edges_by_weight orders weights across the whole float line", "[pairemst]" ) {
        /// The oracle is `std::stable_sort` on the weight alone: the radix sort
        /// promises weight order, not the `(weight, a, b)` tie-break.
        auto sorted_weights = []( std::vector<Edge> edges ) {
            std::vector<Edge> scratch;
            sort_edges_by_weight( edges, scratch );
            std::vector<float> out;
            for ( const Edge& e : edges ) {
                out.push_back( e.weight );
            }
            return out;
        };
        auto expected_weights = []( std::vector<Edge> edges ) {
            std::stable_sort( edges.begin(), edges.end(), []( const Edge& l, const Edge& r ) {
                return l.weight < r.weight;
            } );
            std::vector<float> out;
            for ( const Edge& e : edges ) {
                out.push_back( e.weight );
            }
            return out;
        };
        auto edges_of = []( const std::vector<float>& weights ) {
            std::vector<Edge> edges;
            for ( uint32_t i = 0; i < weights.size(); i++ ) {
                edges.push_back( Edge{ .weight = weights[i], .a = i, .b = i + 1 } );
            }
            return edges;
        };

        SECTION( "a negative weight sorts first, not last" ) {
            const auto edges = edges_of( { 1.0f, 0.5f, -1.19e-7f, 0.0f } );
            REQUIRE( sorted_weights( edges ) == std::vector<float>{ -1.19e-7f, 0.0f, 0.5f, 1.0f } );
        }

        SECTION( "negative zero sorts no later than positive zero" ) {
            const auto out = sorted_weights( edges_of( { 1.0f, 0.0f, -0.0f } ) );
            REQUIRE( out.size() == 3 );
            REQUIRE( std::signbit( out[0] ) );
            REQUIRE( out[2] == 1.0f );
        }

        SECTION( "random weights of both signs and wide magnitude" ) {
            std::mt19937 rng( 42 );
            std::uniform_real_distribution<float> mantissa( -1.0f, 1.0f );
            std::uniform_int_distribution<int> exponent( -20, 20 );
            std::vector<float> weights;
            for ( size_t i = 0; i < 50000; i++ ) {
                weights.push_back( std::ldexp( mantissa( rng ), exponent( rng ) ) );
            }
            const auto edges = edges_of( weights );
            REQUIRE( sorted_weights( edges ) == expected_weights( edges ) );
        }

        SECTION( "shared high bytes exercise the pass skip" ) {
            std::vector<float> weights;
            for ( size_t i = 0; i < 1000; i++ ) {
                weights.push_back( 1.0f + static_cast<float>( ( i * 7919 ) % 1000 ) * 1e-6f );
            }
            const auto edges = edges_of( weights );
            REQUIRE( sorted_weights( edges ) == expected_weights( edges ) );
        }

        SECTION( "trivial sizes" ) {
            REQUIRE( sorted_weights( {} ).empty() );
            REQUIRE( sorted_weights( edges_of( { 3.0f } ) ) == std::vector<float>{ 3.0f } );
        }
    }

    TEST_CASE( "ParallelExceptionGuard carries an exception out of a parallel region",
               "[pairemst]" ) {
        /// Without the guard, a throw escaping the region below would call
        /// `std::terminate` and take the whole test binary down with it.
        auto throw_from_one_iteration = [] {
            ParallelExceptionGuard guard;
#pragma omp parallel for num_threads( 4 ) schedule( dynamic, 1 )
            for ( int i = 0; i < 64; i++ ) {
                guard.run( [i] {
                    if ( i == 37 ) {
                        throw std::runtime_error( "iteration 37" );
                    }
                } );
            }
            guard.rethrow_if_failed();
        };
        REQUIRE_THROWS_AS( throw_from_one_iteration(), std::runtime_error );

        SECTION( "a clean region does not throw" ) {
            /// Catch2 assertions are not thread-safe, so the region only
            /// counts, and the checks happen after it.
            ParallelExceptionGuard guard;
            int sum = 0;
            int failures = 0;
#pragma omp parallel for num_threads( 4 ) reduction( + : sum, failures )
            for ( int i = 0; i < 64; i++ ) {
                if ( !guard.run( [&] { sum += i; } ) ) {
                    failures++;
                }
            }
            REQUIRE_NOTHROW( guard.rethrow_if_failed() );
            REQUIRE( failures == 0 );
            REQUIRE( sum == 64 * 63 / 2 );
        }
    }

} // namespace panna
