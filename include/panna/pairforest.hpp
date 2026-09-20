#pragma once

//! A memory-compact LSH forest specialised for enumerating *colliding pairs*.
//!
//! `panna::Index` (`panna/trieindex.hpp`) keeps, for every repetition, the full
//! hash value next to every point id. The pair-enumeration code paths never
//! read those hash values: all they ask of a repetition is
//!
//!   1. give me the point ids in sorted hash order, and
//!   2. do sorted positions `i` and `j` agree on their first `k` hash symbols?
//!
//! Both questions are answered by `PairCompactTree` from a handful of bitvectors
//! that are derived from the hashes once, at construction time, after which the
//! hashes are thrown away. That brings the per-point, per-repetition cost down
//! from `4 + sizeof(HashValue)` bytes to `4 + K/8 + K/16` bytes.

#include <algorithm>
#include <bit>
#include <cstdint>
#include <functional>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <omp.h>

#include "panna/data.hpp" // Edge
#include "panna/expect.hpp"
#include "panna/logging.hpp"
#include "panna/lsh/predicates.hpp" // failure_probability
#include "panna/timer.hpp"

namespace panna {

    //! The sorted point-id sequence of one LSH repetition, plus enough
    //! bookkeeping to answer "do sorted positions `i` and `j` share their first
    //! `k` hash symbols?" without storing the hash values themselves.
    //!
    //! The hashes are consumed by `assign` to derive the order and the class
    //! boundaries, then discarded by the caller. Storage is `4 + K/8 + K/16`
    //! bytes per point (4.75 B at K = 4) against the `4 + sizeof(HashValue)`
    //! bytes of a `PrefixMap` (36 B with `LongLshValue<4>`).
    //!
    //! The structure is immutable once assigned and every accessor is `const`,
    //! so any number of threads may read it concurrently.
    class PairCompactTree {
    public:
        //! Side of a square tile, in sorted positions. 128 row points and 128
        //! column points of a 100-dimensional int16 dataset occupy ~51 KiB,
        //! which stays resident in L2 for the whole double loop.
        static constexpr uint32_t TILE_SIZE = 128;

        //! A pair of ranges of **sorted positions** (not point ids) that both
        //! lie inside a single bucket at the prefix length the tile was produced
        //! for. Invariants:
        //!   - `row_end - row_begin <= TILE_SIZE`, `col_end - col_begin <= TILE_SIZE`
        //!   - `col_begin >= row_begin`
        //!   - either `row_begin == col_begin && row_end == col_end` (diagonal)
        //!     or `col_begin >= row_end` (disjoint, off-diagonal)
        //! A diagonal tile holds the pairs `{(i, j) : row_begin <= i < j < row_end}`;
        //! an off-diagonal tile holds the full cartesian product of its two ranges.
        struct Tile {
            uint32_t row_begin;
            uint32_t row_end;
            uint32_t col_begin;
            uint32_t col_end;

            //! `true` when the two ranges coincide, i.e. when the caller must
            //! start the inner loop at `i + 1` instead of at `col_begin`.
            bool is_diagonal() const {
                return row_begin == col_begin;
            }
        };

        PairCompactTree() = default;

        //! Builds the tree from `sorted`, which must already be sorted by
        //! `std::sort` over `std::pair<HashValue, uint32_t>` -- hash first, point
        //! id as the tie-break. That is byte-for-byte the ordering produced by
        //! `PrefixMap::rebuild`, so the resulting id sequence is identical
        //! position for position.
        //!
        //! `k_max` is the number of hash symbols, i.e.
        //! `HashValue::get_concatenations()`.
        //!
        //! Throws `std::length_error` if `sorted` holds more than `2^32 - 1`
        //! entries. A real `throw` rather than `expect`, because `expect`
        //! compiles to a no-op unless `EXPECT_ACTIVE` is defined.
        template <typename HashValue>
        void assign( const std::vector<std::pair<HashValue, uint32_t>>& sorted, uint8_t k_max ) {
            if ( sorted.size() > std::numeric_limits<uint32_t>::max() ) {
                throw std::length_error( "PairCompactTree supports at most 2^32-1 points" );
            }

            num_levels = k_max;
            num_points = static_cast<uint32_t>( sorted.size() );
            words_per_level = ( num_points + 63 ) / 64;

            ids.resize( num_points );
            for ( uint32_t i = 0; i < num_points; i++ ) {
                ids[i] = sorted[i].second;
            }

            boundaries.assign( static_cast<size_t>( num_levels ) * words_per_level, 0 );
            rank_directory.assign( static_cast<size_t>( num_levels ) * words_per_level, 0 );
            if ( num_points == 0 ) {
                return;
            }

            // --- pass 1: mark the first position of every class ---------------
            // Position 0 starts a class at every level.
            for ( unsigned k = 1; k <= num_levels; k++ ) {
                set_boundary( static_cast<uint8_t>( k ), 0 );
            }
            for ( uint32_t i = 1; i < num_points; i++ ) {
                /// `prefix_eq` is monotone in the prefix length for any two
                /// values: once two hashes differ at length `d` they differ at
                /// every length >= d. So we find `d` once and mark every level
                /// from `d` upwards, instead of testing each level separately.
                ///
                /// What lexicographic order buys is separate, and is what makes
                /// comparing only *adjacent* entries sufficient: each prefix
                /// class is a contiguous run, so every class start is a position
                /// that differs from the one before it.
                unsigned d = 1;
                while ( d <= num_levels &&
                        sorted[i - 1].first.prefix_eq( sorted[i].first,
                                                       static_cast<uint8_t>( d ) ) ) {
                    d++;
                }
                for ( unsigned k = d; k <= num_levels; k++ ) {
                    set_boundary( static_cast<uint8_t>( k ), i );
                }
            }

            // --- pass 2: exclusive prefix sum of the per-word popcounts -------
            for ( unsigned k = 1; k <= num_levels; k++ ) {
                const size_t base = static_cast<size_t>( k - 1 ) * words_per_level;
                uint32_t accumulated = 0;
                for ( uint32_t w = 0; w < words_per_level; w++ ) {
                    rank_directory[base + w] = accumulated;
                    accumulated += static_cast<uint32_t>( std::popcount( boundaries[base + w] ) );
                }
            }

            /// The bits beyond `num_points` in the last word of each level are
            /// never set by pass 1 and never read by `rank`, so they contribute
            /// zero to every popcount above.
        }

        //! Number of indexed points.
        uint32_t size() const {
            return num_points;
        }

        //! Number of prefix lengths this tree can answer for, i.e. `K`.
        uint8_t num_prefixes() const {
            return num_levels;
        }

        //! The point ids in sorted hash order.
        const std::vector<uint32_t>& sorted_ids() const {
            return ids;
        }

        //! The point id stored at sorted position `pos`.
        uint32_t id_at( uint32_t pos ) const {
            expect( pos < num_points );
            return ids[pos];
        }

        //! One-based index of the class that sorted position `pos` belongs to,
        //! among the classes induced by the first `k` hash symbols.
        //! Non-decreasing in `pos`. Requires `1 <= k <= num_prefixes()`.
        uint32_t rank( uint32_t pos, uint8_t k ) const {
            expect( pos < num_points );
            expect( k >= 1 && k <= num_levels );
            const size_t base = static_cast<size_t>( k - 1 ) * words_per_level;
            const size_t word = pos / 64;
            /// `~0 >> (63 - pos % 64)` has its low `pos % 64 + 1` bits set, so
            /// the popcount below is *inclusive* of `pos` itself. Together with
            /// the exclusive prefix sum in `rank_directory` this counts every
            /// class start at a position `<= pos`, which is the one-based class
            /// index of `pos`.
            const uint64_t mask = ~uint64_t( 0 ) >> ( 63 - pos % 64 );
            return rank_directory[base + word] +
                   static_cast<uint32_t>( std::popcount( boundaries[base + word] & mask ) );
        }

        //! `true` when sorted positions `i` and `j` share their first `k` hash
        //! symbols. `k == 0` is always `true`: every point shares the empty prefix.
        bool prefix_eq( uint32_t i, uint32_t j, uint8_t k ) const {
            if ( k == 0 ) {
                return true;
            }
            return rank( i, k ) == rank( j, k );
        }

        //! Fills `out` with `out[i] == rank(i, k)` for every position, in one
        //! linear scan. Cheaper than calling `rank` per pair when the same level
        //! is probed many times. Costs 4 bytes per point while alive.
        void labels( uint8_t k, std::vector<uint32_t>& out ) const {
            expect( k >= 1 && k <= num_levels );
            out.resize( num_points );
            const size_t base = static_cast<size_t>( k - 1 ) * words_per_level;
            uint32_t current = 0;
            for ( uint32_t w = 0; w < words_per_level; w++ ) {
                const uint64_t bits = boundaries[base + w];
                // The last word may be only partially populated.
                const uint32_t hi = std::min<uint32_t>( 64, num_points - 64 * w );
                for ( uint32_t b = 0; b < hi; b++ ) {
                    current += static_cast<uint32_t>( ( bits >> b ) & 1 );
                    out[64 * w + b] = current;
                }
            }
        }

        //! Calls `visit(tile)` for every tile at prefix length `prefix`.
        //! `visit` returns `false` to stop the enumeration; `for_each_tile` then
        //! returns `false` as well, and `true` when it ran to completion.
        //!
        //! When `skip_prefix` holds a value (in practice `prefix + 1`), a tile
        //! whose whole covered position span lies inside a single class at that
        //! longer prefix is dropped: every one of its pairs was already
        //! enumerated by the previous, longer-prefix sweep.
        template <typename Visitor>
        bool for_each_tile( uint8_t prefix, std::optional<uint8_t> skip_prefix,
                            Visitor&& visit ) const {
            expect( prefix >= 1 && prefix <= num_levels );
            expect( !skip_prefix || ( *skip_prefix >= 1 && *skip_prefix <= num_levels ) );
            if ( num_points == 0 ) {
                return true;
            }

            uint32_t bucket_begin = 0;
            while ( bucket_begin < num_points ) {
                // The class starts at `prefix` are exactly the boundary bits, so
                // the next one after `bucket_begin` ends the current bucket.
                const uint32_t bucket_end = next_boundary( prefix, bucket_begin + 1 );
                if ( bucket_end - bucket_begin >= 2 ) { // a singleton bucket holds no pair
                    /// Tiles are aligned to `bucket_begin` rather than to
                    /// position 0, so a tile never straddles two buckets and
                    /// `row_begin == col_begin` is exactly the diagonal predicate.
                    for ( uint32_t a = bucket_begin; a < bucket_end; a += TILE_SIZE ) {
                        const uint32_t ae = std::min( a + TILE_SIZE, bucket_end );
                        for ( uint32_t b = a; b < bucket_end; b += TILE_SIZE ) {
                            const uint32_t be = std::min( b + TILE_SIZE, bucket_end );
                            if ( a == b && ae - a < 2 ) {
                                continue; // diagonal tile with fewer than 2 points
                            }
                            /// Tile-level novelty skip. The positions the tile
                            /// can touch all lie in `[a, be)` (because
                            /// `a <= b` and `ae <= be`), and `rank` is
                            /// non-decreasing, so equal ranks at the two ends of
                            /// that span force `rank` to be constant on the whole
                            /// span: every pair of the tile already collided at
                            /// `*skip_prefix` and was reported by that sweep.
                            if ( skip_prefix &&
                                 rank( a, *skip_prefix ) == rank( be - 1, *skip_prefix ) ) {
                                continue;
                            }
                            if ( !visit( Tile{ a, ae, b, be } ) ) {
                                return false;
                            }
                        }
                    }
                }
                bucket_begin = bucket_end;
            }
            return true;
        }

        //! Convenience wrapper that materialises the tiles. Intended for tests
        //! and small inputs; the visitor form above avoids the allocation.
        std::vector<Tile> tiles( uint8_t prefix, std::optional<uint8_t> skip_prefix ) const {
            std::vector<Tile> out;
            for_each_tile( prefix, skip_prefix, [&out]( const Tile& tile ) {
                out.push_back( tile );
                return true;
            } );
            return out;
        }

        //! Approximate number of bytes occupied by this tree, heap included.
        size_t memory_usage() const {
            size_t total = sizeof( *this );
            total += ids.size() * sizeof( uint32_t );
            total += boundaries.size() * sizeof( uint64_t );
            total += rank_directory.size() * sizeof( uint32_t );
            return total;
        }

    private:
        uint8_t num_levels = 0;       //!< = K; level `k` lives at row `k - 1`
        uint32_t num_points = 0;      //!< number of indexed points
        uint32_t words_per_level = 0; //!< ceil(num_points / 64), row stride of both arrays below

        std::vector<uint32_t> ids; //!< point ids in sorted hash order, 4 B/point

        //! `num_levels * words_per_level` words. Bit `pos` of row `k - 1` is set
        //! iff `pos` starts a new class at prefix length `k`. One flat allocation
        //! rather than a vector of vectors: K/8 bytes per point.
        std::vector<uint64_t> boundaries;

        //! `num_levels * words_per_level` entries. Entry `w` of row `k - 1` is
        //! the number of set boundary bits strictly before position `64 * w`
        //! (an exclusive prefix sum of popcounts): K/16 bytes per point.
        std::vector<uint32_t> rank_directory;

        //! The smallest position `>= from` that starts a class at prefix length
        //! `k`, or `num_points` when there is none.
        uint32_t next_boundary( uint8_t k, uint32_t from ) const {
            if ( from >= num_points ) {
                return num_points;
            }
            const size_t base = static_cast<size_t>( k - 1 ) * words_per_level;
            uint32_t w = from / 64;
            // Mask off the bits below `from` inside its own word, then scan.
            uint64_t bits = boundaries[base + w] & ( ~uint64_t( 0 ) << ( from % 64 ) );
            while ( bits == 0 ) {
                if ( ++w >= words_per_level ) {
                    return num_points;
                }
                bits = boundaries[base + w];
            }
            return std::min<uint32_t>( 64 * w + static_cast<uint32_t>( std::countr_zero( bits ) ),
                                       num_points );
        }

        void set_boundary( uint8_t k, uint32_t pos ) {
            const size_t base = static_cast<size_t>( k - 1 ) * words_per_level;
            boundaries[base + pos / 64] |= uint64_t( 1 ) << ( pos % 64 );
        }
    };

    //! Selects the self-tuning three-argument `Builder::fit`.
    struct FitToData {
        float delta;
    };

    //! Selects the four-argument `Builder::fit`, which rescales the hash family
    //! around `distance_upper_bound`. Only `E2LSHBuilder` and `LatticeLSHBuilder`
    //! react to it (`Builder::fits_to_distance == true`); for the other families
    //! both overloads are no-ops.
    struct FitToDistance {
        float distance_upper_bound;
        float delta;
    };

    //! An immutable, memory-compact LSH index specialised for enumerating
    //! colliding **pairs**. It borrows the dataset, builds every repetition at
    //! construction time and supports no insertion, no update, no serialisation
    //! and no point queries.
    //!
    //! Unlike `panna::Index` it never keeps the hash values: they are produced,
    //! sorted and thrown away inside the constructor, leaving only the sorted
    //! point ids and the class-boundary bitvectors of `PairCompactTree`.
    //!
    //! Correctness, for a fixed repetition `r`. Write
    //! `P_k = { {u,v} : u != v, h(u) and h(v) agree on their first k symbols }`.
    //!  - `HashValue::operator<` is lexicographic over all `K` symbols, so for
    //!    any `k` the positions sharing a `k`-prefix form a maximal contiguous
    //!    run: one sort groups every `k` at once.
    //!  - `PairCompactTree::rank(., k)` is constant exactly on those runs, hence
    //!    `rank(i,k) == rank(j,k)` iff `i` and `j` share their first `k` symbols.
    //!  - `for_each_tile(k, .)` walks the maximal runs and splits each one into a
    //!    block-upper-triangular set of tiles, so it covers `P_k` exactly once.
    //!  - For `k < K`, `search_pairs` drops the pairs whose `rank(., k+1)` agree,
    //!    which are precisely `P_{k+1} ⊆ P_k`, so it reports `P_k \ P_{k+1}`.
    //!    At `k == K` there is no longer prefix and it reports all of `P_K`.
    //! Sweeping `k = K, K-1, ..., 1` therefore reports every colliding pair
    //! exactly once.
    template <typename Dataset, typename Hasher, typename Distance>
    class PairForestIndex {
    public:
        using PointHandle = typename Dataset::PointHandle;
        using HashValue = typename Hasher::Value;
        using Builder = typename Hasher::Builder;
        using Tile = PairCompactTree::Tile;

        //! Number of concatenated hash symbols per repetition. Taken from the
        //! hasher rather than from a free template parameter, which could
        //! silently disagree with it.
        static constexpr uint8_t K = static_cast<uint8_t>( Hasher::get_concatenations() );
        static_assert( K == Hasher::Value::get_concatenations() );
        static_assert( std::is_same_v<Hasher, typename Hasher::Builder::Output> );

        //! Per-thread scratch reused across `search_pairs` calls. Holding one per
        //! worker avoids recomputing the novelty labels for every prefix.
        //!
        //! The cached labels are keyed on the *tree* they were built from, not
        //! on its repetition number, so one scratch may be shared freely across
        //! several `PairForestIndex` instances: repetition 3 of one index is a
        //! different tree from repetition 3 of another, and the cache sees that.
        struct SearchScratch {
            std::vector<uint32_t> novelty_labels; //!< rank(., cached_level); 4 B/point while alive
            std::vector<Edge> tile_buffer;        //!< edges between two callback hand-offs
            //! The tree `novelty_labels` was built from; `nullptr` until populated.
            //! Trees are owned by the index and never move, so the address is a
            //! stable identity for as long as the scratch could be reused.
            const PairCompactTree* cached_tree = nullptr;
            uint8_t cached_level = 0; //!< 0 means "labels not populated"
        };

        //! Fits `hash_builder` to `points` with the self-tuning `fit`, using the
        //! same default failure probability that `Index::rebuild` uses today.
        PairForestIndex( const Dataset& points, size_t repetitions, Builder hash_builder ):
            PairForestIndex( points,
                             repetitions,
                             std::move( hash_builder ),
                             FitToData{ 0.1f / static_cast<float>( points.size() ) } ) {
        }

        //! Fits with the three-argument, self-tuning `Builder::fit`.
        PairForestIndex( const Dataset& points, size_t repetitions, Builder hash_builder,
                         FitToData fit ):
            dataset( points ), repetitions( repetitions ), builder( std::move( hash_builder ) ) {
            prepare( repetitions );
            if ( trees.empty() || dataset.size() == 0 ) {
                return;
            }
            builder.reset();
            fit_to_data( fit.delta );
            hasher = builder.build( repetitions );
            build_all_repetitions();
        }

        //! Fits with the four-argument `Builder::fit`, rescaling the hash family
        //! around `fit.distance_upper_bound`.
        PairForestIndex( const Dataset& points, size_t repetitions, Builder hash_builder,
                         FitToDistance fit ):
            dataset( points ), repetitions( repetitions ), builder( std::move( hash_builder ) ) {
            prepare( repetitions );
            if ( trees.empty() || dataset.size() == 0 ) {
                return;
            }
            /// `reset()` before `fit` mirrors `emst.hpp`: without it
            /// `LatticeLSHBuilder::fit` early-returns on a non-zero scaling factor.
            builder.reset();
            builder.fit( dataset, fit.distance_upper_bound, repetitions, fit.delta );
            hasher = builder.build( repetitions );
            build_all_repetitions();
        }

        // The index borrows its dataset, so it is copyable/movable but not assignable.
        PairForestIndex( const PairForestIndex& ) = default;
        PairForestIndex( PairForestIndex&& ) = default;
        PairForestIndex& operator=( const PairForestIndex& ) = delete;
        PairForestIndex& operator=( PairForestIndex&& ) = delete;

        //! Number of indexed points.
        size_t num_points() const {
            return dataset.size();
        }

        //! Number of LSH repetitions.
        size_t num_repetitions() const {
            return repetitions;
        }

        //! Number of concatenated hash symbols per repetition.
        static constexpr uint8_t num_concatenations() {
            return K;
        }

        //! The borrowed dataset.
        const Dataset& get_dataset() const {
            return dataset;
        }

        //! The fitted hasher. Throws `std::runtime_error` if the index was built
        //! over an empty dataset or with zero repetitions, in which case no
        //! hasher was ever constructed.
        const Hasher& get_hasher() const {
            require_hasher();
            return *hasher;
        }

        //! The compact tree of `repetition`. Throws `std::out_of_range`.
        const PairCompactTree& tree( size_t repetition ) const {
            return trees.at( repetition );
        }

        //! Human-readable name of the LSH family in use.
        std::string describe_family() const {
            return builder.describe();
        }

        //! Approximate number of bytes occupied by the index, heap included.
        //! The borrowed dataset is *not* counted: the index does not own it.
        size_t memory_usage() const {
            size_t total = sizeof( *this );
            for ( const PairCompactTree& t : trees ) {
                total += t.memory_usage();
            }
            return total;
        }

        //! Probability that a pair at `distance` collides on a single hash symbol.
        float collision_probability( float distance ) const {
            require_hasher();
            return hasher->collision_probability( distance );
        }

        //! Probability of never seeing a pair at `distance` when `rep` out of
        //! `num_repetitions()` repetitions have been probed at `concat` symbols.
        float fail_probability( float distance, size_t concat, size_t rep ) const {
            require_hasher();
            return failure_probability( *hasher, distance, concat, rep, repetitions );
        }

        //! How often the collecting overloads below drain their tile buffer into
        //! the caller's vector. Bounding it matters: with an unbounded buffer the
        //! whole result is built twice over, once in the scratch and once in
        //! `output`. 64Ki edges is 768 KiB -- small beside a short prefix's
        //! output, and large enough that the hand-off costs nothing.
        static constexpr size_t COLLECT_BATCH_EDGES = size_t( 1 ) << 16;

        //! Appends to `output` every pair of points that collide on the first
        //! `prefix` hash symbols of `repetition` but **not** on the first
        //! `prefix + 1`, and whose distance is at most `distance_threshold`.
        //! Each reported `Edge` has `a < b` and `weight` equal to the distance.
        //! Returns the number of distance computations performed.
        //!
        //! `prefix` must be in `[1, K]`, otherwise `std::invalid_argument`;
        //! `repetition` must be a valid repetition, otherwise `std::out_of_range`.
        //! Safe to call concurrently from many threads on different repetitions.
        //! A NaN distance fails the threshold test and is silently dropped.
        //!
        //! This form allocates a `SearchScratch` per call, so it recomputes the
        //! novelty labels -- a linear scan and a `4 * n` byte allocation -- every
        //! time. A caller sweeping many `(repetition, prefix)` pairs, as the EMST
        //! search does, should keep one scratch alive and use the overload below.
        size_t search_pairs( size_t repetition, uint8_t prefix, float distance_threshold,
                             std::vector<Edge>& output ) const {
            SearchScratch scratch;
            return search_pairs( repetition, prefix, distance_threshold, output, scratch );
        }

        //! As above, but reusing the caller's `scratch`: the novelty labels are
        //! then rebuilt only when a different tree or prefix is asked for, and
        //! the tile buffer keeps its capacity across calls.
        size_t search_pairs( size_t repetition, uint8_t prefix, float distance_threshold,
                             std::vector<Edge>& output, SearchScratch& scratch ) const {
            return search_pairs(
                repetition,
                prefix,
                distance_threshold,
                COLLECT_BATCH_EDGES,
                [&output]( std::vector<Edge>& batch ) {
                    output.insert( output.end(), batch.begin(), batch.end() );
                    return false; // never stop early
                },
                scratch );
        }

        //! Batched form: instead of accumulating every edge, hands batches to
        //! `batch_output`, which returns `true` to stop the enumeration early.
        //! `buffer_size` is a soft bound -- a batch may overshoot it by at most
        //! one tile, i.e. `TILE_SIZE * TILE_SIZE` edges. The residual buffer is
        //! handed over only when the enumeration ran to completion.
        size_t search_pairs( size_t repetition, uint8_t prefix, float distance_threshold,
                             size_t buffer_size,
                             const std::function<bool( std::vector<Edge>& )>& batch_output,
                             SearchScratch& scratch ) const {
            if ( prefix == 0 || prefix > K ) {
                throw std::invalid_argument(
                    "PairForestIndex::search_pairs: prefix must be in [1, K]" );
            }
            const PairCompactTree& t = trees.at( repetition );

            /// Pairs colliding at `prefix + 1` were already reported by the
            /// previous, longer-prefix sweep. At `prefix == K` there is no
            /// longer prefix, so nothing is subtracted.
            const std::optional<uint8_t> skip =
                ( prefix < K ) ? std::optional<uint8_t>( static_cast<uint8_t>( prefix + 1 ) )
                               : std::nullopt;

            /// `skip` alone decides whether pairs are filtered. Deriving that
            /// from `novelty_labels != nullptr` instead would be wrong for an
            /// empty tree, where `labels()` leaves a zero-size vector whose
            /// `data()` may legitimately be null.
            const uint32_t* novelty_labels = nullptr;
            if ( skip ) {
                ensure_labels( scratch, t, *skip );
                novelty_labels = scratch.novelty_labels.data();
            }

            size_t distance_count = 0;
            scratch.tile_buffer.clear();

            const bool completed =
                t.for_each_tile( prefix, skip, [&]( const Tile& tile ) -> bool {
                    distance_count +=
                        skip ? evaluate_tile<true>(
                                   t, tile, novelty_labels, distance_threshold, scratch.tile_buffer )
                             : evaluate_tile<false>(
                                   t, tile, nullptr, distance_threshold, scratch.tile_buffer );
                    if ( scratch.tile_buffer.size() >= buffer_size ) {
                        const bool stop = batch_output( scratch.tile_buffer );
                        scratch.tile_buffer.clear();
                        return !stop;
                    }
                    return true;
                } );

            if ( completed && !scratch.tile_buffer.empty() ) {
                batch_output( scratch.tile_buffer );
                scratch.tile_buffer.clear();
            }
            return distance_count;
        }

    private:
        const Dataset& dataset; //!< borrowed, never copied
        size_t repetitions;
        std::vector<PairCompactTree> trees; //!< one per repetition

        //! `mutable` only so that the const `fail_probability` can bind `*hasher`
        //! to the non-const `Hasher&` that `panna::failure_probability` takes.
        //! Construction needs no `mutable`: `build_all_repetitions` is itself a
        //! non-const member, which is what lets it call the non-const
        //! `LatticeLSH::hash` / `CrossPolytope::hash`.
        //!
        //! Nothing mutates the hasher after construction, so the concurrent
        //! `search_pairs` this class advertises is safe -- but that holds only
        //! because `failure_probability` calls const methods on it. A hash
        //! family that cached state in those methods would race here.
        mutable std::optional<Hasher> hasher;
        Builder builder; //!< kept for `describe_family()` and for fitting

        //! Common prologue of the three constructors: reject oversized datasets
        //! before the expensive hashing, and size the per-repetition storage.
        void prepare( size_t num_reps ) {
            if ( dataset.size() > std::numeric_limits<uint32_t>::max() ) {
                throw std::length_error( "PairForestIndex supports at most 2^32-1 points" );
            }
            trees.resize( num_reps );
            /// Every tree starts out well-formed and empty, with all `K` levels
            /// declared. A default-constructed `PairCompactTree` reports zero
            /// levels instead, which would put `rank`/`labels`/`for_each_tile`
            /// out of range for an index built over an empty dataset -- the one
            /// case in which `build_all_repetitions` never runs.
            const std::vector<std::pair<HashValue, uint32_t>> nothing;
            for ( PairCompactTree& t : trees ) {
                t.assign( nothing, K );
            }
        }

        //! Calls the three-argument, self-tuning `Builder::fit`.
        //!
        //! `SimhashBuilder` and `CrossPolytopeBuilder` declare this overload as
        //! `fit( Dataset&, size_t, float )` -- a *mutable* reference, even though
        //! both bodies are empty -- while `E2LSHBuilder` and `LatticeLSHBuilder`
        //! take a `const Dataset&`. We only ever hold a const reference, so we
        //! use the const-correct overload when the builder offers one and fall
        //! back to a `const_cast` otherwise. The cast is safe: no `fit`
        //! implementation writes through that reference.
        void fit_to_data( float delta ) {
            if constexpr ( requires( Builder& b, const Dataset& d ) {
                               b.fit( d, size_t( 0 ), float( 0 ) );
                           } ) {
                builder.fit( dataset, repetitions, delta );
            } else {
                builder.fit( const_cast<Dataset&>( dataset ), repetitions, delta );
            }
        }

        void require_hasher() const {
            if ( !hasher ) {
                throw std::runtime_error(
                    "PairForestIndex: no hasher was built (empty dataset or zero repetitions)" );
            }
        }

        //! Hashes every point in every repetition, sorts each repetition and
        //! packs it into a `PairCompactTree`, discarding the hashes as it goes.
        void build_all_repetitions() {
            Timer timer( "pairforest-build" );
            const size_t n = dataset.size();
            // clang-format off
            LOG_INFO( "msg", "building the pair forest",
                      "points", n,
                      "repetitions", repetitions,
                      "concatenations", static_cast<size_t>( K ) );
            // clang-format on

            // --- phase A: hash everything (the memory high-water mark) --------
            /// `Hasher::hash` produces all `L` values of one point at a time, so
            /// we hash point-major and scatter repetition-major. This mirrors
            /// `Index::rebuild`, which likewise calls a possibly non-const
            /// `hash` on a shared hasher from inside an OpenMP loop.
            std::vector<std::vector<HashValue>> per_repetition( repetitions );
            for ( size_t rep = 0; rep < repetitions; rep++ ) {
                per_repetition[rep].resize( n );
            }

#pragma omp parallel
            {
                std::vector<HashValue> point_hashes;
#pragma omp for schedule( static )
                for ( size_t i = 0; i < n; i++ ) {
                    hasher->hash( dataset[i], point_hashes );
                    for ( size_t rep = 0; rep < repetitions; rep++ ) {
                        per_repetition[rep][i] = point_hashes.at( rep );
                    }
                }
            }

            // --- phase B: sort and pack, freeing each repetition as we go -----
            /// `std::sort` over `std::pair<HashValue, uint32_t>` orders by hash
            /// first and by point id as the tie-break -- byte for byte the
            /// ordering of `PrefixMap::rebuild`, so the id sequence of each tree
            /// matches that of the corresponding `PrefixMap` position by position.
            /// Distinct `rep` iterations touch distinct, pre-sized elements of
            /// `trees` and `per_repetition`, so the loop is race-free.
#pragma omp parallel for schedule( dynamic, 1 )
            for ( size_t rep = 0; rep < repetitions; rep++ ) {
                std::vector<std::pair<HashValue, uint32_t>> tmp;
                tmp.reserve( n );
                for ( size_t i = 0; i < n; i++ ) {
                    tmp.emplace_back( per_repetition[rep][i], static_cast<uint32_t>( i ) );
                }
                // Release the transient hashes as soon as they are copied.
                std::vector<HashValue>().swap( per_repetition[rep] );
                std::sort( tmp.begin(), tmp.end() );
                trees[rep].assign( tmp, K );
            }
        }

        //! The pairwise-distance kernel -- the **only** place in the class that
        //! computes a distance, and the only one that reads the dataset outside
        //! construction. Returns the number of distance computations.
        //!
        //! `FilterNovelty` is a template parameter rather than a runtime flag so
        //! that the novelty test disappears entirely from the inner loop when the
        //! caller does not need it (i.e. at `prefix == K`).
        //!
        //! GEMM seam: this method is the whole of the query path's contact with
        //! the data -- nothing else in either class names `Distance`. A float-GEMM
        //! specialisation would replace exactly this method: it would (a) gather
        //! the row and the column points into two contiguous row-major float
        //! matrices -- `EuclideanPoints` is already float, while
        //! `UnitNormPoints`/`NormedPoints` hold `alignas(32) Int16Chunk` and need
        //! an int16 -> float conversion or an int16 GEMM; (b) produce a 128x128
        //! float inner-product block; (c) turn the inner products into the metric
        //! (`1 - dot` for `CosineDistance`, `|a|^2 + |b|^2 - 2*dot` for the
        //! Euclidean ones); (d) run a post-pass applying the novelty labels, the
        //! diagonal `j > i` mask and the threshold. `JaccardDistance` over
        //! `SparseSets` has no inner-product form and must keep the scalar path,
        //! so any such specialisation has to be opt-in per `(Dataset, Distance)`
        //! pair. The indirection is deliberately *not* introduced here.
        template <bool FilterNovelty>
        size_t evaluate_tile( const PairCompactTree& tree, const Tile& tile,
                              [[maybe_unused]] const uint32_t* novelty_labels,
                              float distance_threshold, std::vector<Edge>& output ) const {
            const uint32_t* ids = tree.sorted_ids().data();
            const bool diagonal = tile.is_diagonal();
            size_t computed = 0;

            for ( uint32_t i = tile.row_begin; i < tile.row_end; i++ ) {
                const uint32_t a = ids[i];
                const PointHandle point_a = dataset[a]; // hoisted out of the inner loop
                uint32_t label_a = 0;
                if constexpr ( FilterNovelty ) {
                    label_a = novelty_labels[i];
                }
                // A diagonal tile must not revisit the lower triangle.
                const uint32_t j_begin = diagonal ? i + 1 : tile.col_begin;

                for ( uint32_t j = j_begin; j < tile.col_end; j++ ) {
                    if constexpr ( FilterNovelty ) {
                        if ( novelty_labels[j] == label_a ) {
                            continue; // already reported at prefix + 1
                        }
                    }
                    const uint32_t b = ids[j];
                    const float distance = Distance::compute( point_a, dataset[b] );
                    computed++;
                    // A NaN distance fails this test and is silently dropped.
                    if ( distance <= distance_threshold ) {
                        output.push_back( Edge{ .weight = distance,
                                                .a = std::min( a, b ),
                                                .b = std::max( a, b ) } );
                    }
                }
            }
            return computed;
        }

        //! Makes sure `scratch.novelty_labels` holds `rank(., level)` of `tree`,
        //! recomputing it only when a different tree or level is asked for.
        void ensure_labels( SearchScratch& scratch, const PairCompactTree& tree,
                            uint8_t level ) const {
            if ( scratch.cached_tree == &tree && scratch.cached_level == level ) {
                return;
            }
            tree.labels( level, scratch.novelty_labels );
            scratch.cached_tree = &tree;
            scratch.cached_level = level;
        }
    };

} // namespace panna
