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
//!
//! The `[mr]` cases do the same for `pair_forest_emst_mutual_reachability`,
//! against `EMST::exact_mutual_reachability_distance_tree` as the oracle.

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <mutex>
#include <omp.h>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "panna/data.hpp"
#include "panna/distance.hpp"
#include "panna/dsu.hpp"
#include "panna/emst.hpp"
#include "panna/emst_common.hpp"
#include "panna/linalg.hpp"
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

    TEST_CASE( "buffer_edges_within_budget shrinks the buffers to fit, and refuses below n",
               "[pairemst]" ) {
        const size_t n = 100000;
        const size_t threads = 8;
        const size_t tile_edges =
            static_cast<size_t>( PairCompactTree::TILE_SIZE ) * PairCompactTree::TILE_SIZE;
        /// Restates the cost model of `buffer_edges_within_budget`, so that a
        /// change to it has to be made in two places on purpose.
        auto bytes_needed = [&]( size_t buffer_edges ) {
            const size_t fixed = n * ( 2 * sizeof( Edge ) + 3 * sizeof( uint32_t ) );
            return threads * ( fixed + 2 * sizeof( Edge ) * ( buffer_edges + tile_edges ) );
        };

        SECTION( "plenty of memory gives the full 10n" ) {
            REQUIRE( buffer_edges_within_budget( n, threads, std::numeric_limits<size_t>::max() ) ==
                     PAIR_EMST_BUFFER_EDGES_PER_POINT * n );
            REQUIRE( buffer_edges_within_budget( n, threads, bytes_needed( 10 * n ) ) == 10 * n );
        }

        SECTION( "a tighter budget gives a smaller buffer that still fits" ) {
            const size_t budget = bytes_needed( 4 * n );
            const size_t edges = buffer_edges_within_budget( n, threads, budget );
            REQUIRE( edges == 4 * n );
            REQUIRE( bytes_needed( edges ) <= budget );
        }

        SECTION( "more threads sharing the budget get smaller buffers" ) {
            const size_t budget = bytes_needed( 10 * n );
            REQUIRE( buffer_edges_within_budget( n, 2 * threads, budget ) < 10 * n );
        }

        SECTION( "exactly n edges per buffer is accepted, one byte less is not" ) {
            const size_t budget = bytes_needed( n );
            REQUIRE( buffer_edges_within_budget( n, threads, budget ) == n );
            REQUIRE_THROWS_AS( buffer_edges_within_budget( n, threads, budget - threads ),
                               std::runtime_error );
        }

        SECTION( "a budget that does not even cover the trees is refused" ) {
            REQUIRE_THROWS_AS( buffer_edges_within_budget( n, threads, 0 ), std::runtime_error );
            REQUIRE_THROWS_AS( buffer_edges_within_budget( n, threads, n ), std::runtime_error );
        }
    }

    TEST_CASE( "available_memory_bytes reports a usable amount", "[pairemst]" ) {
        /// Not a statement about this machine, just that the probe does not
        /// come back with zero on a system where the tests could run at all.
        REQUIRE( available_memory_bytes() > 0 );
    }

    namespace pairemst_test {

        //! The same gaussian points twice over: as the `EuclideanPoints` the
        //! pair search takes, and as the vectors `EMST` is built from, which
        //! is what provides the exact mutual-reachability oracle.
        struct TwinData {
            Dataset points;
            std::vector<std::vector<float>> vectors;
        };

        inline TwinData make_twin( size_t dimensions, size_t n, uint64_t seed ) {
            seed_global_rng( seed );
            TwinData twin{ Dataset( dimensions ), {} };
            for ( size_t i = 0; i < n; i++ ) {
                twin.vectors.push_back( sample_random_normal_vector( dimensions ) );
                twin.points.push_back( twin.vectors.back().begin(), twin.vectors.back().end() );
            }
            return twin;
        }

        //! `EMST::exact_mutual_reachability_distance_tree` over `vectors`. The
        //! `EMST` constructor takes its data by non-const reference, hence the
        //! copy; its own index is never searched, so a few repetitions do.
        inline std::pair<float, std::vector<Edge>>
        exact_mr_tree( std::vector<std::vector<float>> vectors, size_t num_neighbors ) {
            const size_t dimensions = vectors.front().size();
            EMST<Dataset, Hasher, Distance> emst( dimensions, 4, vectors, 0.01f, 0.0f );
            return emst.exact_mutual_reachability_distance_tree( num_neighbors );
        }

        //! A mutual-reachability edge weighs at least as much as the core
        //! distance of either endpoint.
        inline void require_dominates_cores( const PairMrEmstResult& result ) {
            if ( result.core_distances.get_num_neighbors() == 0 ) {
                return;
            }
            size_t violations = 0;
            for ( const Edge& e : result.tree ) {
                const float ca =
                    Distance::to_euclidean( result.core_distances.core_distance( e.a ) );
                const float cb =
                    Distance::to_euclidean( result.core_distances.core_distance( e.b ) );
                if ( e.weight < ca || e.weight < cb ) {
                    violations++;
                }
            }
            REQUIRE( violations == 0 );
        }

    } // namespace pairemst_test

    // -----------------------------------------------------------------------
    // PIN — at epsilon = 0 the mutual-reachability tree must weigh what the
    // exact one does. Approx rather than ==: the oracle computes its distances
    // with `Distance::compute`, the search partly with the GEMM tile kernel,
    // and the two may round differently in the last bits.
    // -----------------------------------------------------------------------
    TEST_CASE( "pair_forest_emst_mutual_reachability matches the exact tree at epsilon=0",
               "[pairemst][mr]" ) {
        using namespace pairemst_test;

        const size_t dimensions = 8;
        const size_t n = 2000;
        const size_t repetitions = 512;
        const TwinData twin = make_twin( dimensions, n, 11 );

        for ( size_t num_neighbors : { 0, 1, 5 } ) {
            INFO( "num_neighbors=" << num_neighbors );
            const auto exact = exact_mr_tree( twin.vectors, num_neighbors );

            const PairMrEmstResult result =
                pair_forest_emst_mutual_reachability<Dataset, Hasher, Distance>(
                    twin.points, num_neighbors, 0.0f, 0.001f, repetitions );

            require_spanning_tree( result.tree, n );
            require_dominates_cores( result );
            CHECK( result.weight == Catch::Approx( exact.first ).epsilon( 1e-5 ) );
            CHECK( tree_weight( result.tree ) == Catch::Approx( exact.first ).epsilon( 1e-5 ) );
            CHECK( result.core_distances.get_num_neighbors() == num_neighbors );
            CHECK( result.distances_computed > 0 );
            CHECK( result.repetitions_at_stop > 0 );

            /// The core estimates of the tree's endpoints are exact at
            /// epsilon = 0, and every point is an endpoint.
            if ( num_neighbors > 0 ) {
                CoreDistances oracle( n, num_neighbors );
                for ( size_t i = 0; i < n; i++ ) {
                    for ( size_t j = 0; j < n; j++ ) {
                        if ( i != j ) {
                            oracle.update_one_sided(
                                static_cast<uint32_t>( i ),
                                static_cast<uint32_t>( j ),
                                Distance::compute( twin.points[i], twin.points[j] ) );
                        }
                    }
                }
                size_t wrong_cores = 0;
                for ( uint32_t i = 0; i < n; i++ ) {
                    if ( result.core_distances.core_distance( i ) !=
                         Catch::Approx( oracle.core_distance( i ) ).epsilon( 1e-5 ) ) {
                        wrong_cores++;
                    }
                }
                CHECK( wrong_cores == 0 );
            }
        }
    }

    // -----------------------------------------------------------------------
    // INVARIANT — the (1 + epsilon) bound, on the mutual-reachability weight.
    // -----------------------------------------------------------------------
    TEST_CASE( "pair_forest_emst_mutual_reachability respects the (1+epsilon) bound",
               "[pairemst][mr]" ) {
        using namespace pairemst_test;

        const size_t dimensions = 8;
        const size_t n = 1000;
        const size_t repetitions = 512;
        const size_t num_neighbors = 5;
        const float epsilon = 0.2f;

        for ( uint64_t seed : { 1, 2, 3 } ) {
            INFO( "seed=" << seed );
            const TwinData twin = make_twin( dimensions, n, seed );
            const float exact_weight = exact_mr_tree( twin.vectors, num_neighbors ).first;
            REQUIRE( exact_weight > 0.0f );

            const PairMrEmstResult result =
                pair_forest_emst_mutual_reachability<Dataset, Hasher, Distance>(
                    twin.points, num_neighbors, epsilon, 0.01f, repetitions );

            require_spanning_tree( result.tree, n );
            require_dominates_cores( result );
            CHECK( result.weight >= exact_weight * ( 1.0f - 1e-5f ) );
            CHECK( result.weight <= ( 1.0f + epsilon ) * exact_weight * ( 1.0f + 1e-5f ) );
        }
    }

    // -----------------------------------------------------------------------
    // Degenerate inputs.
    // -----------------------------------------------------------------------
    TEST_CASE( "pair_forest_emst_mutual_reachability handles degenerate inputs",
               "[pairemst][mr]" ) {
        using namespace pairemst_test;

        auto run = []( const TwinData& twin, size_t num_neighbors ) {
            return pair_forest_emst_mutual_reachability<Dataset, Hasher, Distance>(
                twin.points, num_neighbors, 0.0f, 0.01f, 256 );
        };

        SECTION( "duplicate points" ) {
            // Ten distinct positions, five copies each. With 3 neighbors every
            // core distance is zero, so the tree is the plain one; with 6 it is
            // the distance to the nearest *other* position.
            const size_t dimensions = 4;
            seed_global_rng( 77 );
            TwinData twin{ Dataset( dimensions ), {} };
            std::vector<std::vector<float>> distinct;
            for ( size_t i = 0; i < 10; i++ ) {
                distinct.push_back( sample_random_normal_vector( dimensions ) );
            }
            for ( size_t copy = 0; copy < 5; copy++ ) {
                for ( const auto& p : distinct ) {
                    twin.vectors.push_back( p );
                    twin.points.push_back( p.begin(), p.end() );
                }
            }
            const size_t n = twin.points.size();

            for ( size_t num_neighbors : { 3, 6 } ) {
                INFO( "num_neighbors=" << num_neighbors );
                const float exact_weight = exact_mr_tree( twin.vectors, num_neighbors ).first;
                const PairMrEmstResult result = run( twin, num_neighbors );
                require_spanning_tree( result.tree, n );
                require_dominates_cores( result );
                CHECK( result.weight == Catch::Approx( exact_weight ).epsilon( 1e-5 ) );
            }
        }

        SECTION( "two points" ) {
            const TwinData twin = make_twin( 4, 2, 5 );
            const float d = Distance::compute( twin.points[0], twin.points[1] );
            for ( size_t num_neighbors : { 0, 1 } ) {
                INFO( "num_neighbors=" << num_neighbors );
                const PairMrEmstResult result = run( twin, num_neighbors );
                require_spanning_tree( result.tree, 2 );
                CHECK( result.weight == Catch::Approx( d ) );
            }
        }

        SECTION( "as many neighbors as there are other points" ) {
            // Every core distance is the distance to the farthest point, so
            // every edge weighs about as much as the diameter, and at
            // epsilon = 0 the stopping rule would have to confirm distances
            // that large -- beyond what 256 repetitions can, for this search
            // and for `EMST`'s alike. A loose epsilon lets it stop on the
            // lower bound instead.
            const size_t n = 40;
            const float epsilon = 0.5f;
            const TwinData twin = make_twin( 4, n, 606 );
            const float exact_weight = exact_mr_tree( twin.vectors, n - 1 ).first;
            const PairMrEmstResult result =
                pair_forest_emst_mutual_reachability<Dataset, Hasher, Distance>(
                    twin.points, n - 1, epsilon, 0.01f, 256 );
            require_spanning_tree( result.tree, n );
            require_dominates_cores( result );
            CHECK( result.weight >= exact_weight * ( 1.0f - 1e-5f ) );
            CHECK( result.weight <= ( 1.0f + epsilon ) * exact_weight * ( 1.0f + 1e-5f ) );
        }

        SECTION( "more neighbors than there are points" ) {
            // No point has that many neighbors: every core distance, and so
            // every edge, is infinite, as in the oracle.
            const size_t n = 40;
            const TwinData twin = make_twin( 4, n, 607 );
            const float exact_weight = exact_mr_tree( twin.vectors, n + 3 ).first;
            const PairMrEmstResult result = run( twin, n + 3 );
            require_spanning_tree( result.tree, n );
            CHECK( std::isinf( exact_weight ) );
            CHECK( std::isinf( result.weight ) );
        }

        SECTION( "fewer than two points is rejected" ) {
            const TwinData twin = make_twin( 4, 1, 4 );
            REQUIRE_THROWS_AS( run( twin, 3 ), std::invalid_argument );
        }
    }

    // -----------------------------------------------------------------------
    // RETENTION — the tree must be the minimum spanning tree of every edge the
    // search ever held, under the final core distances, even though it drops
    // most of them along the way. The other [mr] cases cannot see a mistake in
    // the retention rule: with many repetitions, index seeding and NN-descent
    // the cores are exact before the sweep starts, so no weight ever changes.
    // Here the cores start poor -- no seeding, no NN-descent -- so they drop
    // *during* the sweep, which is the case the batch-end neighborhood pass
    // and the routing of evicted pairs exist for. Many threads and whole
    // batches of repetitions make the flushes interleave; clustered data makes
    // cores drop by orders of magnitude, and duplicate points make distances
    // tie. Invariant (N) is also checked directly, after every batch.
    // -----------------------------------------------------------------------
    namespace pairemst_test {

        //! Gaussian clusters with centers spread 10 apart and scales spanning
        //! three orders of magnitude, about 50 points each.
        inline Dataset make_clustered_dataset( size_t dimensions, size_t n, uint64_t seed ) {
            std::mt19937_64 rng( seed );
            std::normal_distribution<float> gaussian( 0.0f, 1.0f );
            std::uniform_real_distribution<float> uniform( 0.0f, 1.0f );
            const size_t num_clusters = std::max<size_t>( 2, n / 50 );
            std::vector<std::vector<float>> centers( num_clusters,
                                                     std::vector<float>( dimensions ) );
            std::vector<float> scales( num_clusters );
            for ( size_t c = 0; c < num_clusters; c++ ) {
                for ( float& x : centers[c] ) {
                    x = 10.0f * gaussian( rng );
                }
                scales[c] = std::pow( 10.0f, -3.0f * uniform( rng ) );
            }
            Dataset data( dimensions );
            std::vector<float> point( dimensions );
            for ( size_t i = 0; i < n; i++ ) {
                const size_t c = rng() % num_clusters;
                for ( size_t j = 0; j < dimensions; j++ ) {
                    point[j] = centers[c][j] + scales[c] * gaussian( rng );
                }
                data.push_back( point.begin(), point.end() );
            }
            return data;
        }

        //! `n` points drawn with replacement from `n / 7` gaussian ones, so
        //! each point has about six exact copies.
        inline Dataset make_duplicated_dataset( size_t dimensions, size_t n, uint64_t seed ) {
            std::mt19937_64 rng( seed );
            std::normal_distribution<float> gaussian( 0.0f, 1.0f );
            const size_t distinct = std::max<size_t>( 2, n / 7 );
            std::vector<std::vector<float>> base( distinct, std::vector<float>( dimensions ) );
            for ( auto& point : base ) {
                for ( float& x : point ) {
                    x = gaussian( rng );
                }
            }
            Dataset data( dimensions );
            for ( size_t i = 0; i < n; i++ ) {
                const auto& point = base[rng() % distinct];
                data.push_back( point.begin(), point.end() );
            }
            return data;
        }

        //! Points of the integer grid `{0, ..., 5}^dimensions`, drawn with
        //! replacement: distances tie all over the place, and in 8 dimensions
        //! a few points coincide. The ties are what give an edge lost by the
        //! batch-end pass a weight no other edge can stand in for.
        inline Dataset make_grid_dataset( size_t dimensions, size_t n, uint64_t seed ) {
            std::mt19937_64 rng( seed );
            Dataset data( dimensions );
            std::vector<float> point( dimensions );
            for ( size_t i = 0; i < n; i++ ) {
                for ( float& x : point ) {
                    x = static_cast<float>( rng() % 6 );
                }
                data.push_back( point.begin(), point.end() );
            }
            return data;
        }

        //! Sets the OpenMP thread count for as long as it lives.
        struct ScopedOmpThreads {
            int saved;
            explicit ScopedOmpThreads( int threads ): saved( omp_get_max_threads() ) {
                omp_set_num_threads( threads );
            }
            ~ScopedOmpThreads() {
                omp_set_num_threads( saved );
            }
            ScopedOmpThreads( const ScopedOmpThreads& ) = delete;
            ScopedOmpThreads& operator=( const ScopedOmpThreads& ) = delete;
        };

    } // namespace pairemst_test

    TEST_CASE( "pair_forest_emst_mutual_reachability loses no edge it needs", "[pairemst][mr]" ) {
        using namespace pairemst_test;

        const size_t dimensions = 8;
        const size_t n = 2000;
        const ScopedOmpThreads threads( 16 );

        enum class Shape { gaussian, clustered, duplicated, grid };
        for ( Shape shape : { Shape::gaussian, Shape::clustered, Shape::duplicated, Shape::grid } ) {
            for ( size_t num_neighbors : { 5, 15 } ) {
                /// A single repetition leaves most pairs found only once, so
                /// the batch-end pass is all that stands between a stored pair
                /// whose weight dropped and its loss. Whole batches rediscover
                /// pairs over and over, which hides that, but make the flushes
                /// of 16 threads interleave.
                for ( size_t repetitions : { 1, 32, 64 } ) {
                    const uint64_t seed = 4242 + repetitions + 7 * num_neighbors;
                    const Dataset data = shape == Shape::gaussian
                                             ? make_dataset( dimensions, n, seed )
                                         : shape == Shape::clustered
                                             ? make_clustered_dataset( dimensions, n, seed )
                                         : shape == Shape::duplicated
                                             ? make_duplicated_dataset( dimensions, n, seed )
                                             : make_grid_dataset( dimensions, n, seed );
                    INFO( "shape=" << static_cast<int>( shape ) << " num_neighbors="
                                   << num_neighbors << " repetitions=" << repetitions );

                    /// Every edge the search could ever have used: the seed
                    /// tree, the neighborhoods it starts from, and every pair a
                    /// flush was handed. Evicted pairs need no list of their
                    /// own, since each one entered a neighborhood through one
                    /// of those.
                    std::vector<Edge> seen;
                    std::mutex seen_mutex;
                    size_t batches_checked = 0;
                    size_t batches_wrong = 0;
                    size_t invariant_violations = 0;
                    double worst_gap = 0.0;

                    /// (N): whenever `d(a, b) < core(a)`, `b` is in `NN(a)`.
                    /// The slack keeps a pair whose two stored copies differ in
                    /// the last bits from counting against a tie at the core.
                    auto count_violations = [&]( const CoreDistances& cores ) {
                        size_t violations = 0;
                        auto holds = [&]( uint32_t p, uint32_t q, float d ) {
                            if ( !( d < cores.core_distance( p ) * ( 1.0f - 1e-6f ) ) ) {
                                return true;
                            }
                            const auto [begin, end] = cores.neighbors_view( p );
                            return std::any_of(
                                begin, end, [&]( const auto& slot ) { return slot.second == q; } );
                        };
                        for ( const Edge& e : seen ) {
                            if ( e.a == e.b ) {
                                continue;
                            }
                            violations += !holds( e.a, e.b, e.weight );
                            violations += !holds( e.b, e.a, e.weight );
                        }
                        return violations;
                    };

                    PairMrEmstHooks hooks;
                    hooks.seed_from_index = false;
                    hooks.on_start = [&]( const std::vector<Edge>& seed_tree,
                                          const CoreDistances& cores ) {
                        seen.insert( seen.end(), seed_tree.begin(), seed_tree.end() );
                        const size_t k = cores.get_num_neighbors();
                        const auto& all = cores.all();
                        for ( size_t i = 0; i < all.size(); i++ ) {
                            if ( all[i].second != std::numeric_limits<uint32_t>::max() ) {
                                seen.push_back( Edge{ .weight = all[i].first,
                                                      .a = static_cast<uint32_t>( i / k ),
                                                      .b = all[i].second } );
                            }
                        }
                        invariant_violations += count_violations( cores );
                    };
                    hooks.on_flush = [&]( const std::vector<Edge>& buffer ) {
                        std::lock_guard<std::mutex> lock( seen_mutex );
                        seen.insert( seen.end(), buffer.begin(), buffer.end() );
                    };
                    hooks.on_batch = [&]( const std::vector<MREdge>& tree,
                                          const CoreDistances& cores ) {
                        invariant_violations += count_violations( cores );

                        std::vector<Edge> weighted;
                        weighted.reserve( seen.size() );
                        for ( const Edge& e : seen ) {
                            weighted.push_back(
                                Edge{ .weight = cores.mutual_reachability_distance( e ),
                                      .a = e.a,
                                      .b = e.b } );
                        }
                        std::sort( weighted.begin(), weighted.end() );
                        DSU dsu( static_cast<uint32_t>( n ) );
                        std::vector<Edge> oracle;
                        kruskal( dsu, weighted, oracle );

                        double oracle_weight = 0.0;
                        for ( const Edge& e : oracle ) {
                            oracle_weight += e.weight;
                        }
                        double tree_weight = 0.0;
                        for ( const MREdge& e : tree ) {
                            tree_weight += e.weight;
                        }
                        /// Absolute on duplicated data, whose tree can weigh
                        /// nothing at all.
                        const double gap =
                            ( tree_weight - oracle_weight ) / std::max( oracle_weight, 1e-12 );
                        worst_gap = std::max( worst_gap, std::abs( gap ) );
                        batches_checked++;
                        batches_wrong += ( oracle.size() != n - 1 || std::abs( gap ) > 1e-6 );
                    };

                    try {
                        pair_forest_emst_mutual_reachability<Dataset, Hasher, Distance>(
                            data, num_neighbors, 10.0f, 0.01f, repetitions, 0, hooks );
                    } catch ( const std::runtime_error& e ) {
                        /// The sweep may run out of repetitions before the
                        /// stopping rule fires; the batches ran all the same.
                        /// Anything else is a real failure.
                        if ( std::string( e.what() ) != "Minimum spanning tree not found" ) {
                            throw;
                        }
                    }
                    INFO( "worst relative gap=" << worst_gap );
                    REQUIRE( batches_checked > 0 );
                    CHECK( invariant_violations == 0 );
                    CHECK( batches_wrong == 0 );
                }
            }
        }
    }

    // The batch-end pass scans only the neighborhoods of the points whose
    // core dropped, and must still emit every stored pair exactly once --
    // above all, a pair stored on both sides must not be left to a side that
    // is not scanned. Random inputs rarely hit that (see above), so it is set
    // up by hand here.
    TEST_CASE( "collect_neighborhood_edges emits each stored pair of a lowered point once",
               "[pairemst][mr]" ) {
        const float inf = std::numeric_limits<float>::infinity();

        CoreDistances cores( 5, 2 );
        cores.update( 0, 1, 1.0f ); // NN(0) = {1, 2}, NN(1) = {0, 3}
        cores.update( 0, 2, 3.0f );
        cores.update( 1, 3, 4.0f );
        SharedCoreDistances shared( std::move( cores ) );

        auto count_pair = []( const std::vector<MREdge>& edges, uint32_t p, uint32_t q ) {
            return std::count_if( edges.begin(), edges.end(), [&]( const MREdge& e ) {
                return ( e.a == p && e.b == q ) || ( e.a == q && e.b == p );
            } );
        };

        // Evicts 3 from NN(1): core(1) drops from 4 to 2, and 1 is lowered.
        // 0 is not, and it stores 1 too, with the smaller id.
        std::vector<Edge> evicted;
        shared.insert( 1, 4, 2.0f, evicted );
        REQUIRE( shared.is_lowered( 1 ) );
        REQUIRE( !shared.is_lowered( 0 ) );

        SECTION( "a pair stored on both sides, only the larger id lowered" ) {
            std::vector<MREdge> out;
            collect_neighborhood_edges( shared, inf, /*only_lowered=*/true, out );
            CHECK( count_pair( out, 0, 1 ) == 1 );
            CHECK( count_pair( out, 1, 4 ) == 1 );
            // NN(0) is not scanned: its pair with 2 did not get lighter.
            CHECK( count_pair( out, 0, 2 ) == 0 );
        }

        SECTION( "a pair stored on both sides, both lowered" ) {
            shared.insert( 0, 3, 0.5f, evicted ); // evicts 2 from NN(0)
            REQUIRE( shared.is_lowered( 0 ) );
            std::vector<MREdge> out;
            collect_neighborhood_edges( shared, inf, /*only_lowered=*/true, out );
            CHECK( count_pair( out, 0, 1 ) == 1 );
            CHECK( count_pair( out, 0, 3 ) == 1 );
        }

        SECTION( "the full scan emits every stored pair once" ) {
            std::vector<MREdge> out;
            collect_neighborhood_edges( shared, inf, /*only_lowered=*/false, out );
            CHECK( count_pair( out, 0, 1 ) == 1 );
            CHECK( count_pair( out, 0, 2 ) == 1 );
            CHECK( count_pair( out, 1, 4 ) == 1 );
            CHECK( count_pair( out, 1, 3 ) == 1 ); // NN(3) still stores 1
            CHECK( out.size() == 4 );
        }
    }

    TEST_CASE( "mr_buffer_edges_within_budget charges the shared state and refuses below n",
               "[pairemst][mr]" ) {
        const size_t n = 100000;
        const size_t threads = 8;

        SECTION( "plenty of memory gives the full 10n" ) {
            REQUIRE( mr_buffer_edges_within_budget(
                         n, 5, threads, std::numeric_limits<size_t>::max() ) ==
                     PAIR_EMST_BUFFER_EDGES_PER_POINT * n );
        }

        SECTION( "more neighbors leave less for the buffers" ) {
            const size_t budget = size_t( 1 ) << 30;
            REQUIRE( mr_buffer_edges_within_budget( n, 50, threads, budget ) <
                     mr_buffer_edges_within_budget( n, 5, threads, budget ) );
        }

        SECTION( "it never gives more than the plain search would" ) {
            const size_t budget = size_t( 2 ) << 30;
            REQUIRE( mr_buffer_edges_within_budget( n, 0, threads, budget ) <=
                     buffer_edges_within_budget( n, threads, budget ) );
        }

        SECTION( "a budget that does not even cover the shared state is refused" ) {
            REQUIRE_THROWS_AS( mr_buffer_edges_within_budget( n, 5, threads, n ),
                               std::runtime_error );
        }

        SECTION( "the end of a batch is not charged on top of the buffers" ) {
            // It runs after the per-thread state is freed. Charging both at
            // once refused this feasible configuration.
            REQUIRE( mr_buffer_edges_within_budget( 5000000, 20, 16, size_t( 18 ) << 30 ) >=
                     PAIR_EMST_MIN_BUFFER_EDGES_PER_POINT * 5000000 );
        }

        SECTION( "an end of batch that does not fit is refused, and says so" ) {
            // 1000 neighbors: 0.8 GB of core distances, but up to 3.2 GB of
            // neighborhood candidates at the end of a batch.
            std::string message;
            try {
                mr_buffer_edges_within_budget( n, 1000, 1, size_t( 2 ) << 30 );
            } catch ( const std::runtime_error& e ) {
                message = e.what();
            }
            REQUIRE( message.find( "end of a batch" ) != std::string::npos );
        }
    }

    TEST_CASE( "clustering_emst seeds a spanning tree when points are zero vectors",
               "[pairemst]" ) {
        // Under the cosine distance a zero vector is at distance 1 from every
        // point, itself included. Once the k-center radius drops below 1 such a
        // point is the farthest one, and if choosing it as a center did not
        // take it out of the running it would be chosen again and again: the
        // duplicate centers then give a "tree" with more than `n - 1` edges.
        const size_t dimensions = 8;
        seed_global_rng( 1234 );
        UnitNormPoints data( dimensions );
        // Tight clusters, so that the radius drops well below 1.
        const std::vector<float> base = sample_random_normal_vector( dimensions );
        for ( size_t i = 0; i < 400; i++ ) {
            std::vector<float> x = sample_random_normal_vector( dimensions );
            for ( size_t d = 0; d < dimensions; d++ ) {
                x[d] = base[d] + 0.01f * x[d];
            }
            normalize( x );
            data.push_back( x.begin(), x.end() );
        }
        const std::vector<float> zero( dimensions, 0.0f );
        for ( size_t i = 0; i < 5; i++ ) {
            data.push_back( zero.begin(), zero.end() );
        }
        const size_t n = data.size();

        const auto clustering = kcenter<CosineDistance>( data, 30 );
        std::vector<size_t> centers = clustering.center_indices;
        std::sort( centers.begin(), centers.end() );
        REQUIRE( std::adjacent_find( centers.begin(), centers.end() ) == centers.end() );

        const std::vector<Edge> seed = clustering_emst<UnitNormPoints, CosineDistance>( data );
        pairemst_test::require_spanning_tree( seed, n );
    }

} // namespace panna
