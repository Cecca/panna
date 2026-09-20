#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <iterator>
#include <limits>
#include <optional>
#include <random>
#include <utility>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "panna/data.hpp"
#include "panna/distance.hpp"
#include "panna/dsu.hpp"
#include "panna/lsh/simhash.hpp"
#include "panna/lsh/values.hpp"
#include "panna/pairforest.hpp"
#include "panna/rand.hpp"
#include "panna/trieindex.hpp"

namespace panna {
    namespace pairforest_test {

        //! A pair of *sorted positions*, which is what the tree enumerates.
        using PositionPair = std::pair<uint32_t, uint32_t>;

        //! A pair of *point ids*, which is what `search_pairs` reports.
        using IdPair = std::pair<uint32_t, uint32_t>;

        template <typename HashValue>
        using SortedHashes = std::vector<std::pair<HashValue, uint32_t>>;

        //! Brute-force oracle for `PairCompactTree::rank`: the one-based index of
        //! the class of position `i`, obtained by counting the prefix changes
        //! between consecutive entries.
        template <typename HashValue>
        uint32_t oracle_rank( const SortedHashes<HashValue>& sorted, uint32_t i, uint8_t k ) {
            uint32_t rank = 1;
            for ( uint32_t t = 1; t <= i; t++ ) {
                if ( !sorted[t - 1].first.prefix_eq( sorted[t].first, k ) ) {
                    rank++;
                }
            }
            return rank;
        }

        //! Brute-force oracle for the colliding position pairs at prefix `k`.
        //! Returned sorted, so it can be compared with `==` against an expansion.
        template <typename HashValue>
        std::vector<PositionPair> oracle_pairs( const SortedHashes<HashValue>& sorted,
                                                uint8_t k ) {
            std::vector<PositionPair> out;
            const uint32_t n = static_cast<uint32_t>( sorted.size() );
            for ( uint32_t i = 0; i < n; i++ ) {
                for ( uint32_t j = i + 1; j < n; j++ ) {
                    if ( sorted[i].first.prefix_eq( sorted[j].first, k ) ) {
                        out.emplace_back( i, j );
                    }
                }
            }
            std::sort( out.begin(), out.end() );
            return out;
        }

        //! `a \ b`, both inputs sorted.
        std::vector<PositionPair> difference( const std::vector<PositionPair>& a,
                                              const std::vector<PositionPair>& b ) {
            std::vector<PositionPair> out;
            std::set_difference(
                a.begin(), a.end(), b.begin(), b.end(), std::back_inserter( out ) );
            return out;
        }

        //! Random hash values over the alphabet `[low, low + alphabet)`, sorted
        //! exactly the way `PrefixMap::rebuild` (and hence `PairForestIndex`)
        //! sorts them: by hash value first, by point id as the tie-break.
        template <uint8_t K>
        SortedHashes<IntLshValue<K>> make_sorted_hashes( size_t n, int32_t alphabet, uint64_t seed,
                                                         int32_t low = 0 ) {
            std::mt19937_64 rng( seed );
            std::uniform_int_distribution<int32_t> symbol( low, low + alphabet - 1 );
            SortedHashes<IntLshValue<K>> out;
            out.reserve( n );
            for ( size_t i = 0; i < n; i++ ) {
                std::array<int32_t, K> symbols{};
                for ( uint8_t k = 0; k < K; k++ ) {
                    symbols[k] = symbol( rng );
                }
                out.emplace_back( IntLshValue<K>::make( symbols ), static_cast<uint32_t>( i ) );
            }
            std::sort( out.begin(), out.end() );
            return out;
        }

        //! Expands tiles into the position pairs they stand for, applying the
        //! diagonal rule. The result is sorted; duplicates are preserved so that
        //! the caller can detect them.
        std::vector<PositionPair> expand_tiles( const std::vector<PairCompactTree::Tile>& tiles ) {
            std::vector<PositionPair> out;
            for ( const auto& tile : tiles ) {
                for ( uint32_t i = tile.row_begin; i < tile.row_end; i++ ) {
                    const uint32_t j_begin = tile.is_diagonal() ? i + 1 : tile.col_begin;
                    for ( uint32_t j = j_begin; j < tile.col_end; j++ ) {
                        out.emplace_back( i, j );
                    }
                }
            }
            std::sort( out.begin(), out.end() );
            return out;
        }

        //! Keeps only the pairs whose labels differ, i.e. the per-pair novelty
        //! filter that `evaluate_tile<true>` applies.
        std::vector<PositionPair> apply_label_filter( const std::vector<PositionPair>& pairs,
                                                      const std::vector<uint32_t>& labels ) {
            std::vector<PositionPair> out;
            for ( const auto& p : pairs ) {
                if ( labels[p.first] != labels[p.second] ) {
                    out.push_back( p );
                }
            }
            return out;
        }

        std::vector<IdPair> edge_ids( const std::vector<Edge>& edges ) {
            std::vector<IdPair> out;
            out.reserve( edges.size() );
            for ( const Edge& e : edges ) {
                out.emplace_back( e.a, e.b );
            }
            std::sort( out.begin(), out.end() );
            return out;
        }

        // =====================================================================
        // 1. PairCompactTree: rank and prefix_eq on a hand-written nesting
        // =====================================================================

        TEST_CASE( "PairCompactTree rank and prefix_eq", "[pairforest]" ) {
            using Value = ShortLshValue<4>;

            // Deliberately nested: the five values agree on longer and longer
            // prefixes as we walk backwards from the last one.
            const std::vector<std::array<int16_t, 4>> raw = {
                { 1, 1, 1, 1 }, { 1, 1, 1, 2 }, { 1, 1, 2, 0 }, { 1, 3, 0, 0 }, { 2, 0, 0, 0 }
            };

            SortedHashes<Value> sorted;
            for ( size_t i = 0; i < raw.size(); i++ ) {
                // Ids deliberately not equal to the positions.
                sorted.emplace_back( Value::make( raw[i] ), static_cast<uint32_t>( 10 + i ) );
            }
            REQUIRE( std::is_sorted( sorted.begin(), sorted.end() ) );

            PairCompactTree tree;
            tree.assign( sorted, 4 );

            REQUIRE( tree.size() == 5 );
            REQUIRE( tree.num_prefixes() == 4 );
            for ( uint32_t i = 0; i < 5; i++ ) {
                REQUIRE( tree.id_at( i ) == 10 + i );
            }
            REQUIRE( tree.sorted_ids() == std::vector<uint32_t>{ 10, 11, 12, 13, 14 } );

            // Explicit expectations, spelled out so the oracle itself is checked.
            // k = 1: 1 1 1 1 2            -> 1 1 1 1 2
            // k = 2: 11 11 11 13 20       -> 1 1 1 2 3
            // k = 3: 111 111 112 130 200  -> 1 1 2 3 4
            // k = 4: all distinct         -> 1 2 3 4 5
            const std::vector<std::vector<uint32_t>> expected_ranks = { { 1, 1, 1, 1, 2 },
                                                                       { 1, 1, 1, 2, 3 },
                                                                       { 1, 1, 2, 3, 4 },
                                                                       { 1, 2, 3, 4, 5 } };
            for ( uint8_t k = 1; k <= 4; k++ ) {
                for ( uint32_t i = 0; i < 5; i++ ) {
                    REQUIRE( tree.rank( i, k ) == expected_ranks[k - 1][i] );
                    REQUIRE( tree.rank( i, k ) == oracle_rank( sorted, i, k ) );
                }
            }

            for ( uint8_t k = 1; k <= 4; k++ ) {
                for ( uint32_t i = 0; i < 5; i++ ) {
                    for ( uint32_t j = 0; j < 5; j++ ) {
                        REQUIRE( tree.prefix_eq( i, j, k ) ==
                                 sorted[i].first.prefix_eq( sorted[j].first, k ) );
                    }
                }
            }

            // Every point shares the empty prefix.
            for ( uint32_t i = 0; i < 5; i++ ) {
                for ( uint32_t j = 0; j < 5; j++ ) {
                    REQUIRE( tree.prefix_eq( i, j, 0 ) );
                }
            }
        }

        // =====================================================================
        // 2. PairCompactTree: word boundaries and the labels/rank agreement
        // =====================================================================

        TEST_CASE( "PairCompactTree word boundaries", "[pairforest]" ) {
            constexpr uint8_t K = 4;
            // 63/64/65 straddle a 64-bit word; 127/128/129 straddle a tile.
            const std::vector<size_t> sizes = { 0, 1, 2, 63, 64, 65, 127, 128, 129, 1000 };

            // Three shapes: a small non-negative alphabet (so buckets are
            // non-trivial at every level), an alphabet with negative symbols
            // (the lexicographic order is signed), and all-identical hashes.
            struct Shape {
                const char* name;
                int32_t alphabet;
                int32_t low;
            };
            const std::vector<Shape> shapes = {
                { "small alphabet", 3, 0 }, { "signed alphabet", 5, -2 }, { "all identical", 1, 0 }
            };

            for ( const Shape& shape : shapes ) {
                for ( size_t n : sizes ) {
                    INFO( "shape=" << shape.name << " n=" << n );
                    const auto sorted =
                        make_sorted_hashes<K>( n, shape.alphabet, 0xC0FFEEull + n, shape.low );
                    REQUIRE( std::is_sorted( sorted.begin(), sorted.end() ) );

                    PairCompactTree tree;
                    tree.assign( sorted, K );
                    REQUIRE( tree.size() == n );
                    REQUIRE( tree.num_prefixes() == K );

                    // Large loops accumulate a mismatch count instead of firing
                    // one Catch2 assertion per element.
                    size_t rank_mismatches = 0;
                    size_t label_mismatches = 0;
                    std::vector<uint32_t> labels;
                    for ( uint8_t k = 1; k <= K; k++ ) {
                        tree.labels( k, labels );
                        REQUIRE( labels.size() == n );
                        for ( uint32_t i = 0; i < n; i++ ) {
                            const uint32_t expected = oracle_rank( sorted, i, k );
                            if ( tree.rank( i, k ) != expected ) {
                                rank_mismatches++;
                            }
                            if ( labels[i] != expected ) {
                                label_mismatches++;
                            }
                        }
                    }
                    REQUIRE( rank_mismatches == 0 );
                    REQUIRE( label_mismatches == 0 );

                    size_t prefix_eq_mismatches = 0;
                    for ( uint8_t k = 1; k <= K; k++ ) {
                        for ( uint32_t i = 0; i < n; i++ ) {
                            for ( uint32_t j = 0; j < n; j++ ) {
                                const bool expected =
                                    sorted[i].first.prefix_eq( sorted[j].first, k );
                                if ( tree.prefix_eq( i, j, k ) != expected ) {
                                    prefix_eq_mismatches++;
                                }
                            }
                        }
                    }
                    REQUIRE( prefix_eq_mismatches == 0 );
                }
            }

            // Storage is 4 + K/8 + K/16 bytes per point, plus a constant.
            const auto sorted = make_sorted_hashes<K>( 1000, 3, 7 );
            PairCompactTree tree;
            tree.assign( sorted, K );
            const double expected_bytes = 1000.0 * ( 4.0 + K / 8.0 + K / 16.0 );
            REQUIRE( tree.memory_usage() >= 0.95 * expected_bytes );
            REQUIRE( tree.memory_usage() <= 1.10 * expected_bytes );
        }

        // =====================================================================
        // 3. PairCompactTree: the tiles cover every colliding pair exactly once
        // =====================================================================

        TEST_CASE( "PairCompactTree tiles cover each pair exactly once", "[pairforest]" ) {
            constexpr uint8_t K = 4;
            const std::vector<size_t> sizes = { 0, 1, 2, 63, 64, 65, 127, 128, 129, 300, 1000 };

            for ( size_t n : sizes ) {
                for ( int32_t alphabet : { 1, 3 } ) { // 1 = all hashes identical
                    INFO( "n=" << n << " alphabet=" << alphabet );
                    const auto sorted = make_sorted_hashes<K>( n, alphabet, 0xBEEFull + n );
                    PairCompactTree tree;
                    tree.assign( sorted, K );

                    for ( uint8_t k = 1; k <= K; k++ ) {
                        const auto tiles = tree.tiles( k, std::nullopt );

                        size_t shape_violations = 0;
                        for ( const auto& tile : tiles ) {
                            const bool ok =
                                tile.row_end - tile.row_begin <= PairCompactTree::TILE_SIZE &&
                                tile.col_end - tile.col_begin <= PairCompactTree::TILE_SIZE &&
                                tile.col_begin >= tile.row_begin &&
                                ( tile.is_diagonal() ? ( tile.row_end == tile.col_end )
                                                     : ( tile.col_begin >= tile.row_end ) );
                            if ( !ok ) {
                                shape_violations++;
                            }
                        }
                        REQUIRE( shape_violations == 0 );

                        auto expanded = expand_tiles( tiles );
                        const size_t with_duplicates = expanded.size();
                        expanded.erase( std::unique( expanded.begin(), expanded.end() ),
                                        expanded.end() );
                        REQUIRE( expanded.size() == with_duplicates ); // no pair emitted twice
                        REQUIRE( expanded == oracle_pairs( sorted, k ) );
                    }
                }
            }

            // 300 identical hashes form a single bucket of 300 at every level,
            // which splits into exactly 3 blocks, i.e. 3*4/2 = 6 tiles.
            const auto identical = make_sorted_hashes<K>( 300, 1, 1 );
            PairCompactTree tree;
            tree.assign( identical, K );
            for ( uint8_t k = 1; k <= K; k++ ) {
                REQUIRE( tree.tiles( k, std::nullopt ).size() == 6 );
            }
        }

        // =====================================================================
        // 4. PairCompactTree: the tile-level novelty skip is sound
        // =====================================================================

        TEST_CASE( "PairCompactTree tile novelty skip is sound", "[pairforest]" ) {
            constexpr uint8_t K = 4;
            const std::vector<size_t> sizes = { 0, 1, 2, 64, 129, 300, 1000 };

            for ( size_t n : sizes ) {
                for ( int32_t alphabet : { 1, 3 } ) {
                    INFO( "n=" << n << " alphabet=" << alphabet );
                    const auto sorted = make_sorted_hashes<K>( n, alphabet, 0xF00Dull + n );
                    PairCompactTree tree;
                    tree.assign( sorted, K );

                    for ( uint8_t k = 1; k < K; k++ ) {
                        const uint8_t next = static_cast<uint8_t>( k + 1 );
                        std::vector<uint32_t> labels;
                        tree.labels( next, labels );

                        const auto skipped_tiles = tree.tiles( k, next );
                        const auto all_tiles = tree.tiles( k, std::nullopt );
                        REQUIRE( skipped_tiles.size() <= all_tiles.size() );

                        // `<=` alone would also hold for a skip that never
                        // fires, which would leave the optimisation untested.
                        // This pins the all-or-nothing end of the range; the
                        // partial case is pinned in its own case below.
                        if ( alphabet == 1 ) {
                            // Every hash is identical, so the whole repetition
                            // is one class at `next`: every tile is old work.
                            REQUIRE( skipped_tiles.empty() );
                        }

                        const auto expected =
                            difference( oracle_pairs( sorted, k ), oracle_pairs( sorted, next ) );

                        const auto from_skipped =
                            apply_label_filter( expand_tiles( skipped_tiles ), labels );
                        const auto from_all =
                            apply_label_filter( expand_tiles( all_tiles ), labels );

                        // The tile-level skip only saves work: same output.
                        REQUIRE( from_skipped == from_all );
                        REQUIRE( from_skipped == expected );
                    }
                }
            }
        }

        //! The skip drops a tile only when the whole tile fits inside one
        //! `(k+1)` class, so it fires only once the buckets are several tiles
        //! wide. A 2-symbol alphabet over 2000 points gives exactly that
        //! regime at every level: some tiles sit inside a class and are
        //! dropped, others straddle a boundary and are kept. Without this, a
        //! skip predicate that never fired would still pass every other
        //! assertion in the suite.
        TEST_CASE( "PairCompactTree tile novelty skip actually fires", "[pairforest]" ) {
            constexpr uint8_t K = 4;
            const auto sorted = make_sorted_hashes<K>( 2000, 2, 0xBEEFull );
            PairCompactTree tree;
            tree.assign( sorted, K );

            for ( uint8_t k = 1; k < K; k++ ) {
                INFO( "k=" << static_cast<int>( k ) );
                const auto all_tiles = tree.tiles( k, std::nullopt );
                const auto skipped_tiles = tree.tiles( k, static_cast<uint8_t>( k + 1 ) );

                REQUIRE( !all_tiles.empty() );
                REQUIRE( !skipped_tiles.empty() );                  // not everything dropped
                REQUIRE( skipped_tiles.size() < all_tiles.size() ); // something dropped
            }
        }

        // =====================================================================
        // 5. PairForestIndex reproduces the pair sets of panna::Index
        // =====================================================================

        TEST_CASE( "PairForestIndex matches Index on the same pair sets", "[pairforest]" ) {
            constexpr uint8_t K = 24;
            using Hasher = Simhash<K, UnitNormPoints, CosineDistance>;
            using IndexT = Index<UnitNormPoints, Hasher, CosineDistance>;
            using ForestT = PairForestIndex<UnitNormPoints, Hasher, CosineDistance>;

            const size_t dimensions = 32;
            const size_t n = 1000;
            const size_t repetitions = 4;
            const float inf = std::numeric_limits<float>::infinity();

            seed_global_rng( 9001 );
            std::vector<std::vector<float>> raw;
            for ( size_t i = 0; i < n; i++ ) {
                raw.push_back( sample_random_normal_vector( dimensions ) );
            }

            SimhashBuilder<K, UnitNormPoints, CosineDistance> builder( dimensions );

            IndexT index( dimensions, builder, repetitions );
            for ( const auto& p : raw ) {
                index.insert( p.begin(), p.end() );
            }
            // Both structures draw their random projections inside `build`, so
            // reseeding immediately before each construction makes them agree.
            seed_global_rng( 1234 );
            index.rebuild();

            UnitNormPoints dataset( dimensions );
            for ( const auto& p : raw ) {
                dataset.push_back( p.begin(), p.end() );
            }
            seed_global_rng( 1234 );
            ForestT forest( dataset, repetitions, builder );

            REQUIRE( forest.num_points() == n );
            REQUIRE( forest.num_repetitions() == repetitions );
            REQUIRE( ForestT::num_concatenations() == K );

            size_t total_reported = 0;
            for ( size_t rep = 0; rep < repetitions; rep++ ) {
                for ( uint8_t prefix = 1; prefix <= K; prefix++ ) {
                    INFO( "rep=" << rep << " prefix=" << +prefix );
                    std::vector<Edge> from_index;
                    DSU dsu( static_cast<uint32_t>( n ) ); // nothing connected: nothing skipped
                    index.search_pairs_filter( rep, prefix, from_index, inf, dsu );

                    std::vector<Edge> from_forest;
                    forest.search_pairs( rep, prefix, inf, from_forest );

                    REQUIRE( from_index.size() == from_forest.size() );
                    REQUIRE( edge_ids( from_index ) == edge_ids( from_forest ) );

                    std::sort( from_index.begin(), from_index.end() );
                    std::sort( from_forest.begin(), from_forest.end() );
                    size_t weight_mismatches = 0;
                    for ( size_t i = 0; i < from_index.size(); i++ ) {
                        if ( from_forest[i].weight != Catch::Approx( from_index[i].weight ) ) {
                            weight_mismatches++;
                        }
                    }
                    REQUIRE( weight_mismatches == 0 );
                    total_reported += from_forest.size();
                }
            }
            // Guards against a vacuously green cross-check: with 1000 points and
            // 24 random bits the shortest prefixes must report plenty of pairs.
            REQUIRE( total_reported > n );
        }

        // =====================================================================
        // 6. PairForestIndex::search_pairs against a brute-force oracle
        // =====================================================================

        TEST_CASE( "PairForestIndex search_pairs matches brute force", "[pairforest]" ) {
            constexpr uint8_t K = 8;
            using Hasher = Simhash<K, UnitNormPoints, CosineDistance>;
            using ForestT = PairForestIndex<UnitNormPoints, Hasher, CosineDistance>;

            const size_t dimensions = 16;
            const size_t n = 500;
            const size_t repetitions = 4;
            const float inf = std::numeric_limits<float>::infinity();

            seed_global_rng( 4242 );
            UnitNormPoints dataset( dimensions );
            for ( size_t i = 0; i < n; i++ ) {
                dataset.push_back_random();
            }

            SimhashBuilder<K, UnitNormPoints, CosineDistance> builder( dimensions );
            ForestT forest( dataset, repetitions, builder );

            // `Simhash::hash` is const, so the index's own hasher can rebuild the
            // oracle without any extra randomness.
            std::vector<std::vector<typename Hasher::Value>> hashes( n );
            for ( size_t i = 0; i < n; i++ ) {
                forest.get_hasher().hash( dataset[i], hashes[i] );
            }

            auto oracle = [&]( size_t rep, uint8_t k ) {
                std::vector<IdPair> out;
                for ( uint32_t a = 0; a < n; a++ ) {
                    for ( uint32_t b = a + 1; b < n; b++ ) {
                        if ( hashes[a][rep].prefix_eq( hashes[b][rep], k ) ) {
                            out.emplace_back( a, b );
                        }
                    }
                }
                std::sort( out.begin(), out.end() );
                return out;
            };

            for ( size_t rep = 0; rep < repetitions; rep++ ) {
                std::vector<IdPair> union_over_k;
                std::vector<Edge> all_edges;

                for ( uint8_t k = K; k >= 1; k-- ) {
                    INFO( "rep=" << rep << " k=" << +k );
                    std::vector<Edge> edges;
                    const size_t computed = forest.search_pairs( rep, k, inf, edges );
                    REQUIRE( computed >= edges.size() );

                    size_t orientation_errors = 0;
                    size_t weight_errors = 0;
                    for ( const Edge& e : edges ) {
                        if ( e.a >= e.b ) {
                            orientation_errors++;
                        }
                        const float expected =
                            CosineDistance::compute( dataset[e.a], dataset[e.b] );
                        if ( e.weight != Catch::Approx( expected ) ) {
                            weight_errors++;
                        }
                    }
                    REQUIRE( orientation_errors == 0 );
                    REQUIRE( weight_errors == 0 );

                    // The reported set is exactly P_k \ P_{k+1} (and P_K at k = K).
                    std::vector<IdPair> expected = oracle( rep, k );
                    if ( k < K ) {
                        const auto longer = oracle( rep, static_cast<uint8_t>( k + 1 ) );
                        std::vector<IdPair> diff;
                        std::set_difference( expected.begin(),
                                             expected.end(),
                                             longer.begin(),
                                             longer.end(),
                                             std::back_inserter( diff ) );
                        expected.swap( diff );
                    }
                    REQUIRE( edge_ids( edges ) == expected );

                    union_over_k.insert( union_over_k.end(), expected.begin(), expected.end() );
                    all_edges.insert( all_edges.end(), edges.begin(), edges.end() );
                }

                // No pair is reported twice over the whole descending sweep, and
                // the union is exactly the set of pairs colliding on one symbol.
                std::sort( union_over_k.begin(), union_over_k.end() );
                const size_t with_duplicates = union_over_k.size();
                union_over_k.erase( std::unique( union_over_k.begin(), union_over_k.end() ),
                                    union_over_k.end() );
                REQUIRE( union_over_k.size() == with_duplicates );
                REQUIRE( union_over_k == oracle( rep, 1 ) );
                REQUIRE( union_over_k.size() > n ); // the sweep is not vacuous

                // Thresholding: the median reported weight must cut the output
                // down to exactly the edges that pass the test.
                std::vector<float> weights;
                for ( const Edge& e : all_edges ) {
                    weights.push_back( e.weight );
                }
                REQUIRE( !weights.empty() );
                std::sort( weights.begin(), weights.end() );
                const float threshold = weights[weights.size() / 2];

                size_t threshold_mismatches = 0;
                for ( uint8_t k = K; k >= 1; k-- ) {
                    std::vector<Edge> unfiltered;
                    forest.search_pairs( rep, k, inf, unfiltered );
                    std::vector<Edge> expected;
                    for ( const Edge& e : unfiltered ) {
                        if ( e.weight <= threshold ) {
                            expected.push_back( e );
                        }
                    }
                    std::vector<Edge> filtered;
                    forest.search_pairs( rep, k, threshold, filtered );
                    if ( edge_ids( filtered ) != edge_ids( expected ) ) {
                        threshold_mismatches++;
                    }
                }
                REQUIRE( threshold_mismatches == 0 );
            }
        }

        // =====================================================================
        // 7. PairForestIndex degenerate inputs
        // =====================================================================

        TEST_CASE( "PairForestIndex degenerate inputs", "[pairforest]" ) {
            constexpr uint8_t K = 8;
            using Hasher = Simhash<K, UnitNormPoints, CosineDistance>;
            using ForestT = PairForestIndex<UnitNormPoints, Hasher, CosineDistance>;

            const size_t dimensions = 16;
            const float inf = std::numeric_limits<float>::infinity();
            SimhashBuilder<K, UnitNormPoints, CosineDistance> builder( dimensions );

            seed_global_rng( 20250920 );

            SECTION( "empty dataset" ) {
                UnitNormPoints empty( dimensions );
                REQUIRE_NOTHROW( ForestT( empty, 4, builder ) );

                ForestT forest( empty, 4, builder );
                REQUIRE( forest.num_points() == 0 );
                REQUIRE( forest.num_repetitions() == 4 );
                REQUIRE( forest.tree( 0 ).size() == 0 );
                // Even with nothing to hash, every tree declares all K levels,
                // so the level-indexed accessors stay in range.
                REQUIRE( forest.tree( 0 ).num_prefixes() == K );
                REQUIRE( forest.tree( 0 ).tiles( K, std::nullopt ).empty() );
                // No hasher was ever built, so these must report rather than crash.
                REQUIRE_THROWS_AS( forest.get_hasher(), std::runtime_error );
                REQUIRE_THROWS_AS( forest.collision_probability( 0.5f ), std::runtime_error );
                REQUIRE_THROWS_AS( forest.fail_probability( 0.5f, 1, 1 ), std::runtime_error );

                for ( uint8_t prefix = 1; prefix <= K; prefix++ ) {
                    std::vector<Edge> out;
                    REQUIRE( forest.search_pairs( 0, prefix, inf, out ) == 0 );
                    REQUIRE( out.empty() );
                }
            }

            SECTION( "single point" ) {
                UnitNormPoints one( dimensions );
                one.push_back_random();
                ForestT forest( one, 4, builder );
                REQUIRE( forest.num_points() == 1 );
                REQUIRE( forest.tree( 0 ).size() == 1 );
                for ( uint8_t k = 1; k <= K; k++ ) {
                    REQUIRE( forest.tree( 0 ).rank( 0, k ) == 1 );
                    REQUIRE( forest.tree( 0 ).tiles( k, std::nullopt ).empty() );
                    std::vector<Edge> out;
                    REQUIRE( forest.search_pairs( 0, k, inf, out ) == 0 );
                    REQUIRE( out.empty() );
                }
            }

            SECTION( "zero repetitions" ) {
                UnitNormPoints points( dimensions );
                for ( size_t i = 0; i < 10; i++ ) {
                    points.push_back_random();
                }
                ForestT forest( points, 0, builder );
                REQUIRE( forest.num_repetitions() == 0 );
                std::vector<Edge> out;
                REQUIRE_THROWS_AS( forest.search_pairs( 0, 1, inf, out ), std::out_of_range );
            }

            SECTION( "prefix out of range" ) {
                UnitNormPoints points( dimensions );
                for ( size_t i = 0; i < 10; i++ ) {
                    points.push_back_random();
                }
                ForestT forest( points, 2, builder );
                std::vector<Edge> out;
                REQUIRE_THROWS_AS( forest.search_pairs( 0, 0, inf, out ), std::invalid_argument );
                REQUIRE_THROWS_AS( forest.search_pairs( 0, K + 1, inf, out ),
                                   std::invalid_argument );
                // The prefix is validated before the repetition.
                REQUIRE_THROWS_AS( forest.search_pairs( 99, 0, inf, out ), std::invalid_argument );
                REQUIRE_THROWS_AS( forest.search_pairs( 99, 1, inf, out ), std::out_of_range );
            }

            SECTION( "duplicate points" ) {
                const size_t n = 300;
                std::vector<float> value = sample_random_normal_vector( dimensions );
                UnitNormPoints points( dimensions );
                for ( size_t i = 0; i < n; i++ ) {
                    points.push_back( value.begin(), value.end() );
                }
                ForestT forest( points, 2, builder );

                // Identical points hash identically, so they sit in one class at
                // every level: nothing is new below the longest prefix.
                for ( uint8_t k = 1; k < K; k++ ) {
                    std::vector<Edge> out;
                    forest.search_pairs( 0, k, inf, out );
                    REQUIRE( out.empty() );
                }

                std::vector<Edge> out;
                forest.search_pairs( 0, K, inf, out );
                REQUIRE( out.size() == n * ( n - 1 ) / 2 );

                auto ids = edge_ids( out );
                const size_t with_duplicates = ids.size();
                ids.erase( std::unique( ids.begin(), ids.end() ), ids.end() );
                REQUIRE( ids.size() == with_duplicates );

                /// Duplicated points are at distance ~0, but not at *exactly*
                /// 0: `UnitNormPoints` stores 16-bit fixed point and
                /// `dot_product` accumulates in `int16_t`, so a point's dot
                /// product with its own copy comes out just under 1. We assert
                /// both that the weight is that value and that it is tiny.
                const float self_distance = CosineDistance::compute( points[0], points[1] );
                REQUIRE( self_distance == Catch::Approx( 0.0f ).margin( 1e-3 ) );
                size_t weight_errors = 0;
                for ( const Edge& e : out ) {
                    if ( e.weight != Catch::Approx( self_distance ) ) {
                        weight_errors++;
                    }
                }
                REQUIRE( weight_errors == 0 );
            }
        }

        // =====================================================================
        // 8. The batched overload agrees with the accumulating one
        // =====================================================================

        TEST_CASE( "PairForestIndex batched search_pairs agrees with the simple form",
                   "[pairforest]" ) {
            constexpr uint8_t K = 8;
            using Hasher = Simhash<K, UnitNormPoints, CosineDistance>;
            using ForestT = PairForestIndex<UnitNormPoints, Hasher, CosineDistance>;

            const size_t dimensions = 16;
            const size_t n = 500;
            const size_t repetitions = 2;
            const float inf = std::numeric_limits<float>::infinity();

            seed_global_rng( 4242 );
            UnitNormPoints dataset( dimensions );
            for ( size_t i = 0; i < n; i++ ) {
                dataset.push_back_random();
            }
            SimhashBuilder<K, UnitNormPoints, CosineDistance> builder( dimensions );
            ForestT forest( dataset, repetitions, builder );

            ForestT::SearchScratch scratch;
            for ( size_t rep = 0; rep < repetitions; rep++ ) {
                for ( uint8_t k = 1; k <= K; k++ ) {
                    INFO( "rep=" << rep << " k=" << +k );
                    std::vector<Edge> simple;
                    const size_t simple_count = forest.search_pairs( rep, k, inf, simple );

                    std::vector<Edge> batched;
                    size_t batches = 0;
                    const size_t batched_count = forest.search_pairs(
                        rep,
                        k,
                        inf,
                        64,
                        [&]( std::vector<Edge>& batch ) {
                            batches++;
                            batched.insert( batched.end(), batch.begin(), batch.end() );
                            return false; // never stop early
                        },
                        scratch );

                    REQUIRE( batched_count == simple_count );
                    std::sort( simple.begin(), simple.end() );
                    std::sort( batched.begin(), batched.end() );
                    REQUIRE( simple == batched );
                    if ( !simple.empty() ) {
                        REQUIRE( batches >= 1 );
                    }
                }
            }

            // Returning `true` from the callback stops the enumeration: fewer
            // edges and fewer distance computations than the full run.
            std::vector<Edge> full;
            const size_t full_count = forest.search_pairs( 0, 1, inf, full );
            REQUIRE( full.size() > 64 );

            std::vector<Edge> truncated;
            const size_t truncated_count = forest.search_pairs(
                0,
                1,
                inf,
                64,
                [&]( std::vector<Edge>& batch ) {
                    truncated.insert( truncated.end(), batch.begin(), batch.end() );
                    return true; // stop after the first batch
                },
                scratch );
            REQUIRE( truncated.size() < full.size() );
            REQUIRE( truncated_count < full_count );
        }

        // =====================================================================
        // 9. Compile-time coverage of the supported (Dataset, Distance) pairs
        // =====================================================================

        TEST_CASE( "PairForestIndex instantiates for every supported dataset", "[pairforest]" ) {
            constexpr uint8_t K = 6;
            const size_t dimensions = 8;
            const size_t n = 40;
            const size_t repetitions = 2;
            const float inf = std::numeric_limits<float>::infinity();

            seed_global_rng( 31337 );

            SECTION( "UnitNormPoints with CosineDistance" ) {
                using Hasher = Simhash<K, UnitNormPoints, CosineDistance>;
                using ForestT = PairForestIndex<UnitNormPoints, Hasher, CosineDistance>;

                UnitNormPoints data( dimensions );
                for ( size_t i = 0; i < n; i++ ) {
                    data.push_back_random();
                }
                SimhashBuilder<K, UnitNormPoints, CosineDistance> builder( dimensions );
                ForestT forest( data, repetitions, builder );

                REQUIRE( forest.num_points() == n );
                REQUIRE( forest.describe_family() == "Simhash" );
                REQUIRE( forest.memory_usage() > 0 );
                REQUIRE( forest.collision_probability( 0.0f ) == Catch::Approx( 1.0f ) );
                REQUIRE( forest.fail_probability( 0.0f, 1, 1 ) == Catch::Approx( 0.0f ) );
                REQUIRE( &forest.get_dataset() == &data );

                std::vector<Edge> out;
                forest.search_pairs( 0, K, inf, out );
            }

            SECTION( "NormedPoints with EuclideanDistance" ) {
                using Hasher = Simhash<K, NormedPoints, EuclideanDistance>;
                using ForestT = PairForestIndex<NormedPoints, Hasher, EuclideanDistance>;

                NormedPoints data( dimensions );
                for ( size_t i = 0; i < n; i++ ) {
                    data.push_back_random();
                }
                SimhashBuilder<K, NormedPoints, EuclideanDistance> builder( dimensions );
                // Also exercises the distance-targeted `fit` overload.
                ForestT forest( data, repetitions, builder, FitToDistance{ 1.0f, 0.01f } );

                REQUIRE( forest.num_points() == n );
                REQUIRE( forest.tree( 0 ).size() == n );

                std::vector<Edge> out;
                forest.search_pairs( 0, K, inf, out );
            }

            SECTION( "EuclideanPoints with EuclideanDistance" ) {
                using Hasher = Simhash<K, EuclideanPoints, EuclideanDistance>;
                using ForestT = PairForestIndex<EuclideanPoints, Hasher, EuclideanDistance>;

                EuclideanPoints data( dimensions );
                for ( size_t i = 0; i < n; i++ ) {
                    data.push_back_random();
                }
                SimhashBuilder<K, EuclideanPoints, EuclideanDistance> builder( dimensions );
                ForestT forest( data, repetitions, builder, FitToData{ 0.01f } );

                REQUIRE( forest.num_points() == n );
                REQUIRE( forest.tree( 0 ).size() == n );

                std::vector<Edge> out;
                forest.search_pairs( 0, K, inf, out );
            }
        }

        // =====================================================================
        // 10. PairForestIndex: a SearchScratch may be shared across indices
        // =====================================================================

        //! Regression test. The novelty-label cache used to be keyed on
        //! `(repetition, level)`, so a scratch reused across two indices served
        //! the first index's labels to the second and silently returned the
        //! wrong pair set. The key is the tree's address, which is unique.
        TEST_CASE( "PairForestIndex scratch is not shared between indices", "[pairforest]" ) {
            constexpr uint8_t K = 8;
            using Hasher = Simhash<K, UnitNormPoints, CosineDistance>;
            using ForestT = PairForestIndex<UnitNormPoints, Hasher, CosineDistance>;

            const size_t dimensions = 16;
            const size_t n = 500;
            const size_t repetitions = 2;
            const uint8_t prefix = 3; // < K, so the novelty labels are built
            const float inf = std::numeric_limits<float>::infinity();

            auto make_dataset = []( size_t dims, size_t count, uint64_t seed ) {
                seed_global_rng( seed );
                UnitNormPoints data( dims );
                for ( size_t i = 0; i < count; i++ ) {
                    data.push_back_random();
                }
                return data;
            };

            const UnitNormPoints data_a = make_dataset( dimensions, n, 11111 );
            const UnitNormPoints data_b = make_dataset( dimensions, n, 22222 );

            seed_global_rng( 777 );
            SimhashBuilder<K, UnitNormPoints, CosineDistance> builder_a( dimensions );
            ForestT forest_a( data_a, repetitions, builder_a );

            seed_global_rng( 888 );
            SimhashBuilder<K, UnitNormPoints, CosineDistance> builder_b( dimensions );
            ForestT forest_b( data_b, repetitions, builder_b );

            // Each index queried on its own, with a scratch nothing else touched.
            std::vector<Edge> fresh_a, fresh_b;
            forest_a.search_pairs( 0, prefix, inf, fresh_a );
            forest_b.search_pairs( 0, prefix, inf, fresh_b );

            // The two indices must actually disagree, or the test is vacuous.
            REQUIRE( fresh_a.size() != fresh_b.size() );

            // Now the same scratch for both, same repetition and same prefix --
            // the exact collision the old cache key could not see.
            typename ForestT::SearchScratch shared;
            std::vector<Edge> shared_a, shared_b;
            forest_a.search_pairs(
                0, prefix, inf, std::numeric_limits<size_t>::max(),
                [&shared_a]( std::vector<Edge>& batch ) {
                    shared_a.insert( shared_a.end(), batch.begin(), batch.end() );
                    return false;
                },
                shared );
            forest_b.search_pairs(
                0, prefix, inf, std::numeric_limits<size_t>::max(),
                [&shared_b]( std::vector<Edge>& batch ) {
                    shared_b.insert( shared_b.end(), batch.begin(), batch.end() );
                    return false;
                },
                shared );

            REQUIRE( shared_a.size() == fresh_a.size() );
            REQUIRE( shared_b.size() == fresh_b.size() );
            REQUIRE( shared_a == fresh_a );
            REQUIRE( shared_b == fresh_b );
        }

        // =====================================================================
        // 11. PairForestIndex: the scratch-reusing collecting overload
        // =====================================================================

        //! The collecting overload that takes a caller-supplied `SearchScratch`
        //! must report exactly what the allocate-per-call form reports, for
        //! every repetition and prefix, including when one scratch is carried
        //! across the whole sweep. That reuse is the point: it is what lets the
        //! EMST search pay for the novelty labels once per (tree, prefix)
        //! instead of once per call.
        TEST_CASE( "PairForestIndex search_pairs reuses a caller scratch", "[pairforest]" ) {
            constexpr uint8_t K = 8;
            using Hasher = Simhash<K, UnitNormPoints, CosineDistance>;
            using ForestT = PairForestIndex<UnitNormPoints, Hasher, CosineDistance>;

            const size_t dimensions = 16;
            const size_t n = 500;
            const size_t repetitions = 3;
            const float inf = std::numeric_limits<float>::infinity();

            seed_global_rng( 31337 );
            UnitNormPoints dataset( dimensions );
            for ( size_t i = 0; i < n; i++ ) {
                dataset.push_back_random();
            }
            SimhashBuilder<K, UnitNormPoints, CosineDistance> builder( dimensions );
            ForestT forest( dataset, repetitions, builder );

            // One scratch for the entire sweep, deliberately crossing both
            // repetitions and prefixes so the label cache is invalidated and
            // repopulated many times over.
            typename ForestT::SearchScratch shared;
            size_t total_reported = 0;

            for ( size_t rep = 0; rep < repetitions; rep++ ) {
                for ( uint8_t prefix = K; prefix >= 1; prefix-- ) {
                    INFO( "rep=" << rep << " prefix=" << static_cast<int>( prefix ) );

                    std::vector<Edge> fresh;
                    const size_t fresh_distances =
                        forest.search_pairs( rep, prefix, inf, fresh );

                    std::vector<Edge> reused;
                    const size_t reused_distances =
                        forest.search_pairs( rep, prefix, inf, reused, shared );

                    REQUIRE( reused_distances == fresh_distances );
                    REQUIRE( reused == fresh );
                    total_reported += fresh.size();
                }
            }

            // Guard against the whole case passing vacuously on empty output.
            REQUIRE( total_reported > n );

            // The overload appends rather than clearing, like its sibling.
            std::vector<Edge> appended;
            appended.push_back( Edge{ -1.0f, 0, 0 } );
            forest.search_pairs( 0, K, inf, appended, shared );
            REQUIRE( appended.size() > 1 );
            REQUIRE( appended.front() == Edge{ -1.0f, 0, 0 } );
        }

    } // namespace pairforest_test
} // namespace panna
