#pragma once

//! An approximate Euclidean minimum spanning tree search driven by
//! `PairForestIndex`.
//!
//! `EMST::find_tree` (`panna/emst.hpp`) merges one partial tree *per
//! repetition* into a single collector-owned tree, over a persistent thread
//! pool wired with a `Billboard` and two `Channel`s. The collector is serial,
//! and it is the bottleneck. This search keeps the algorithm and throws that
//! machinery away:
//!
//!  - **one spanning tree per thread, not per repetition.** A thread picks up
//!    repetitions from an OpenMP loop and folds all of them into a single
//!    `local_tree`, so the number of Kruskal merges is governed by how much
//!    edge data flows through, not by how many repetitions there are;
//!  - **edges pile up before they are merged.** A thread buffers the output of
//!    many tiles -- and, within a repetition, of every tile -- and pays for one
//!    Kruskal merge over the lot instead of one per tile;
//!  - **everything is frozen for the duration of a batch.** The input tree, the
//!    confirmed components and therefore the pruning cutoff are read-only
//!    inputs to a batch. No thread ever observes another thread's edges, so
//!    there is no shared mutable state in the hot loop and no synchronisation
//!    beyond the loop's own scheduling;
//!  - **every published tree is already spanning.** Each thread's `local_tree`
//!    starts life as a copy of the frozen input tree, so a thread can only ever
//!    improve a spanning tree into another spanning tree. Merging any subset of
//!    the published trees therefore yields a spanning tree again, and the
//!    minimum spanning forest of the union is preserved whatever order the
//!    reduction happens in.
//!
//! The index is built **once**. `PairForestIndex` is immutable, so there is
//! deliberately no multiscale rehash here, unlike `find_tree`'s sweep over
//! `find_breaks`.

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <fstream>
#include <functional>
#include <limits>
#include <omp.h>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "panna/data.hpp"
#include "panna/dsu.hpp"
#include "panna/emst_common.hpp"
#include "panna/logging.hpp"
#include "panna/pairforest.hpp"
#include "panna/timer.hpp"

namespace panna {

    //! What `pair_forest_emst` returns.
    struct PairEmstResult {
        std::vector<Edge> tree;     //!< sorted ascending, n-1 edges
        float weight;               //!< sum of Distance::to_euclidean(e.weight)
        size_t distances_computed;  //!< distance evaluations over the whole run
        size_t prefix_at_stop;      //!< the prefix length the stopping rule fired at
        size_t repetitions_at_stop; //!< repetitions probed at that prefix
        //! Heap footprint of the index the search built, the dataset excluded.
        //! Reported here because the index is internal to the search and there
        //! is no other way for a caller to measure it without paying to build
        //! a second one.
        size_t index_bytes;
    };

    //! Repetitions are processed in batches of this many. The batch is the unit
    //! over which the input tree and the confirmed components stay frozen, so
    //! it trades two things off: a larger batch means fewer stopping checks and
    //! fewer reductions, a smaller one means the cutoff and the confirmed
    //! components are refreshed sooner and prune harder.
    static constexpr size_t PAIR_EMST_BATCH_REPETITIONS = 32;

    //! Soft bound, in edges, on how much a thread buffers before it folds the
    //! lot into its `local_tree` with one Kruskal merge. Expressed per point
    //! because that is what the merge itself costs: a merge touches the whole
    //! `n - 1` edge tree, so buffering O(n) edges keeps the merge amortised.
    //! This is the size the buffers get when memory allows it; see
    //! `buffer_edges_within_budget` for when it does not.
    static constexpr size_t PAIR_EMST_BUFFER_EDGES_PER_POINT = 10;

    //! The smallest buffer, again in edges per point, worth running with. Below
    //! one tree's worth of edges per flush the merge -- which walks the whole
    //! tree -- costs more than the edges it folds in, and the search would
    //! crawl rather than fail. It is better to refuse to start.
    static constexpr size_t PAIR_EMST_MIN_BUFFER_EDGES_PER_POINT = 1;

    //! Fraction of the available memory the per-thread state may take. The
    //! rest is headroom for what the estimate in `buffer_edges_within_budget`
    //! leaves out: allocator overhead, the reduction, and the rest of the
    //! process.
    static constexpr double PAIR_EMST_MEMORY_FRACTION = 0.8;

    //! The outcome of one evaluation of the stopping rule.
    struct PairEmstStop {
        StoppingConditionInfo info; //!< the confirmed / still-to-confirm split
        float weight_lower_bound;   //!< a lower bound on the true MST weight
        bool should_stop;           //!< the tree is within (1 + epsilon) of optimal
    };

    //! How many OpenMP threads to run a batch of `width` repetitions with.
    //!
    //! There is no point in more threads than there are repetitions to hand
    //! out. `PANNA_EMST_THREADS` overrides the count, exactly as it does for
    //! `EMST::get_worker_count`, so the two searches can be compared under the
    //! same fan-out.
    static size_t pair_emst_worker_count( size_t width ) {
        const size_t available = std::max<size_t>( 1, omp_get_max_threads() );
        size_t workers = std::min( available, std::max<size_t>( 1, width ) );

        if ( const char* env = std::getenv( "PANNA_EMST_THREADS" ); env != nullptr ) {
            char* end = nullptr;
            const unsigned long parsed = std::strtoul( env, &end, 10 );
            if ( end != env && *end == '\0' && parsed > 0 ) {
                workers = std::min( static_cast<size_t>( parsed ), std::max<size_t>( 1, width ) );
            }
        }

        return std::max<size_t>( 1, workers );
    }

    //! The pruning cutoff implied by a spanning tree.
    //!
    //! By the cycle property, adding to a spanning tree any edge heavier than
    //! its heaviest edge creates a cycle on which that new edge is the unique
    //! maximum, so the edge is in no minimum spanning tree of the augmented
    //! graph. A pair farther apart than `tree.back().weight` can therefore
    //! never improve `tree`, and discarding it is safe.
    //!
    //! The slack -- relative `1e-5` plus absolute `1e-7`, inherited from the
    //! prototype this search is modelled on -- exists because the comparison is
    //! made against a distance that was itself computed in floating point,
    //! possibly in a different accumulation order or with different FMA
    //! contraction than when it entered the tree. Without the slack an edge
    //! could be rejected by a comparison against a few-ULP-rounded copy of
    //! itself. It costs a handful of extra distance evaluations.
    static float cutoff_from( const std::vector<Edge>& tree ) {
        if ( tree.empty() ) {
            return std::numeric_limits<float>::infinity();
        }
        return tree.back().weight * ( 1.0f + 1e-5f ) + 1e-7f;
    }

    //! Bytes this process can still allocate without pushing the machine into
    //! swap: `MemAvailable` from `/proc/meminfo`, further capped by the memory
    //! limit of the process's own cgroup (cgroup v2), which is what binds in a
    //! container or a SLURM job. Where neither can be read -- not Linux, say --
    //! the answer is "unbounded", and the buffers get their full size.
    static size_t available_memory_bytes() {
        size_t available = std::numeric_limits<size_t>::max();

        std::ifstream meminfo( "/proc/meminfo" );
        for ( std::string line; std::getline( meminfo, line ); ) {
            if ( line.rfind( "MemAvailable:", 0 ) == 0 ) {
                std::istringstream fields( line.substr( std::strlen( "MemAvailable:" ) ) );
                size_t kilobytes = 0;
                if ( fields >> kilobytes ) {
                    available = kilobytes * 1024;
                }
                break;
            }
        }

        /// Under cgroup v2, `/proc/self/cgroup` is the single line `0::<path>`.
        /// A `memory.max` of `max` means no limit, and fails to parse as a
        /// number, which leaves `available` alone.
        std::ifstream cgroup( "/proc/self/cgroup" );
        std::string line;
        if ( std::getline( cgroup, line ) && line.rfind( "0::", 0 ) == 0 ) {
            const std::string dir = "/sys/fs/cgroup" + line.substr( 3 );
            std::ifstream max_file( dir + "/memory.max" );
            std::ifstream current_file( dir + "/memory.current" );
            size_t limit = 0;
            size_t current = 0;
            if ( max_file >> limit && current_file >> current ) {
                available = std::min( available, ( limit > current ) ? limit - current : 0 );
            }
        }

        return available;
    }

    //! How many edges each of `threads` threads may buffer between two flushes
    //! if their state has to fit in `budget_bytes` all together.
    //!
    //! Per thread, the state that stays alive through a batch is
    //!
    //!  - `local_tree` and `merged`, the two halves of the merge, `n - 1` edges
    //!    each;
    //!  - the merge's `DSU`, two `uint32_t` per point;
    //!  - the novelty labels in `SearchScratch`, one `uint32_t` per point;
    //!  - the edge buffer and the radix sort's ping-pong copy of it, each one
    //!    tile (`TILE_SIZE^2` edges) larger than the buffer itself, since
    //!    `search_pairs` checks the size only after appending a whole tile.
    //!
    //! (Not counted: the GEMM tile scratch in `SearchScratch`, at most ~320 KiB
    //! per thread whatever `n`, and only for `EuclideanPoints`.)
    //!
    //! Only the last item depends on the buffer size, so the budget is spent on
    //! the rest first and whatever remains goes to the buffers, up to
    //! `PAIR_EMST_BUFFER_EDGES_PER_POINT * n` edges.
    //!
    //! Throws `std::runtime_error` when not even
    //! `PAIR_EMST_MIN_BUFFER_EDGES_PER_POINT * n` edges fit.
    static size_t buffer_edges_within_budget( size_t n, size_t threads, size_t budget_bytes ) {
        const size_t tile_edges =
            static_cast<size_t>( PairCompactTree::TILE_SIZE ) * PairCompactTree::TILE_SIZE;
        const size_t fixed_bytes = n * ( 2 * sizeof( Edge ) + 3 * sizeof( uint32_t ) );
        const size_t bytes_per_buffered_edge = 2 * sizeof( Edge );

        const size_t wanted = PAIR_EMST_BUFFER_EDGES_PER_POINT * n;
        const size_t minimum = PAIR_EMST_MIN_BUFFER_EDGES_PER_POINT * n;

        const size_t per_thread_bytes = budget_bytes / std::max<size_t>( 1, threads );
        const size_t buffer_bytes =
            ( per_thread_bytes > fixed_bytes ) ? per_thread_bytes - fixed_bytes : 0;
        const size_t affordable_with_overshoot = buffer_bytes / bytes_per_buffered_edge;
        const size_t affordable =
            ( affordable_with_overshoot > tile_edges ) ? affordable_with_overshoot - tile_edges : 0;

        if ( affordable < minimum ) {
            throw std::runtime_error(
                "pair_forest_emst: not enough memory for the per-thread edge buffers: " +
                std::to_string( threads ) + " threads need at least " +
                std::to_string( threads * ( fixed_bytes +
                                            bytes_per_buffered_edge * ( minimum + tile_edges ) ) ) +
                " bytes, but the memory budget is only " + std::to_string( budget_bytes ) +
                " bytes" );
        }
        return std::min( wanted, affordable );
    }

    //! Step 1: the spanning tree the search starts from.
    //!
    //! Starting from a real spanning tree rather than from an empty forest is
    //! what gives the very first repetition a finite cutoff to prune with, and
    //! it is what makes every intermediate result of the search spanning.
    template <typename Dataset, typename Distance>
    static std::vector<Edge> seed_tree( const Dataset& data ) {
        return clustering_emst<Dataset, Distance>( data );
    }

    //! Step 2: build the index, once.
    //!
    //! Hash families that can be rescaled around a distance (`E2LSH`,
    //! `LatticeLSH`) are fitted to the heaviest edge of the seed tree, which
    //! upper-bounds every edge the search can still be interested in. The
    //! others ignore the distance and self-tune to the data instead.
    //!
    //! `delta_per_pair` is the per-pair failure probability: the union bound
    //! over the `n - 1` edges a spanning tree has.
    template <typename Dataset, typename Hasher, typename Distance>
    static PairForestIndex<Dataset, Hasher, Distance>
    build_index( const Dataset& data,
                 typename Hasher::Builder builder,
                 size_t repetitions,
                 const std::vector<Edge>& initial_tree,
                 float delta_per_pair ) {
        using ForestIndex = PairForestIndex<Dataset, Hasher, Distance>;
        if constexpr ( Hasher::Builder::fits_to_distance ) {
            return ForestIndex( data,
                                repetitions,
                                std::move( builder ),
                                FitToDistance{ initial_tree.back().weight, delta_per_pair } );
        } else {
            return ForestIndex(
                data, repetitions, std::move( builder ), FitToData{ delta_per_pair } );
        }
    }

    //! Sorts `edges` into non-decreasing **weight** order with a four-pass LSD
    //! radix sort, using `scratch` as the ping-pong buffer (it is resized, and
    //! its contents on entry are ignored).
    //!
    //! Why not `std::sort`. A profile of this search on fashion-mnist put 42%
    //! of all cycles inside `std::sort` over `Edge`: a batch flushes hundreds
    //! of thousands of 12-byte records at a time, far more than fits in cache,
    //! and a comparison sort pays `log2(m)` cache-missing passes over them. The
    //! radix sort pays four linear ones, and it is what makes the flush cheaper
    //! than the distance computations that feed it.
    //!
    //! Why sorting on the weight alone is enough. Kruskal is correct for *any*
    //! order that is non-decreasing in the weight; breaking ties differently
    //! only picks a different minimum spanning tree of the same weight. Nothing
    //! downstream needs the `(weight, a, b)` tie-break of `Edge::operator<`:
    //! `kruskal_merge` merges two weight-ordered runs into a weight-ordered
    //! one, and `stopping_condition` walks a weight-ordered prefix. The final
    //! tree is put into full `operator<` order once, at the end of the search.
    //!
    //! Why the float bits can be used as the key. For non-negative IEEE-754
    //! floats the bit pattern read as a `uint32_t` is already monotone in the
    //! value, but for negative ones it is *reversed*, and every negative
    //! pattern compares above every positive one. `key` therefore flips all
    //! bits of a negative float and only the sign bit of a non-negative one,
    //! which makes the unsigned order match the float order on the whole line.
    //! No `Distance` should produce a negative weight, but `CosineDistance`
    //! (`1 - dot`) and `EuclideanDistanceNoSqrt` do no clamping, and rounding
    //! can push either a hair below zero; without the flip such an edge would
    //! silently sort *last* and Kruskal would build a non-minimum tree. `-0.0`
    //! lands just below `+0.0`, which is harmless. A NaN would still sort to one
    //! end; `evaluate_tile` already drops NaN distances, which fail its `<=`
    //! threshold test.
    static void sort_edges_by_weight( std::vector<Edge>& edges, std::vector<Edge>& scratch ) {
        const size_t m = edges.size();
        if ( m < 2 ) {
            return;
        }

        auto key = []( const Edge& e ) -> uint32_t {
            uint32_t bits = 0;
            std::memcpy( &bits, &e.weight, sizeof( bits ) );
            const uint32_t sign_mask = ( bits & 0x80000000u ) ? 0xFFFFFFFFu : 0x80000000u;
            return bits ^ sign_mask;
        };

        /// All four histograms are built in a single pass over the data, so the
        /// records are read five times in total rather than eight.
        uint32_t counts[4][256] = {};
        for ( const Edge& e : edges ) {
            const uint32_t k = key( e );
            counts[0][k & 0xff]++;
            counts[1][( k >> 8 ) & 0xff]++;
            counts[2][( k >> 16 ) & 0xff]++;
            counts[3][( k >> 24 ) & 0xff]++;
        }

        scratch.resize( m );
        std::vector<Edge>* src = &edges;
        std::vector<Edge>* dst = &scratch;

        for ( unsigned pass = 0; pass < 4; pass++ ) {
            const unsigned shift = 8 * pass;
            /// When every key agrees on this byte its histogram has a single
            /// non-zero bucket holding all `m` records, and the pass would be a
            /// plain copy. Distances share an exponent range, so in practice
            /// the two high passes are usually skipped outright.
            const uint8_t first = static_cast<uint8_t>( ( key( ( *src )[0] ) >> shift ) & 0xff );
            if ( counts[pass][first] == m ) {
                continue;
            }

            uint32_t offset[256];
            uint32_t running = 0;
            for ( unsigned b = 0; b < 256; b++ ) {
                offset[b] = running;
                running += counts[pass][b];
            }
            for ( const Edge& e : *src ) {
                const uint8_t b = static_cast<uint8_t>( ( key( e ) >> shift ) & 0xff );
                ( *dst )[offset[b]++] = e;
            }
            std::swap( src, dst );
        }

        /// An odd number of passes leaves the result in `scratch`; swapping is
        /// what puts it back in `edges`, and hands the stale buffer to
        /// `scratch` for the next flush to reuse.
        if ( src != &edges ) {
            edges.swap( scratch );
        }
    }

    //! Folds one buffer of freshly enumerated edges into `local_tree`.
    //!
    //! This is where the batching pays off: `buffer` holds the edges of many
    //! tiles, and they cost a single Kruskal merge between them.
    //!
    //! `search_pairs` reports each pair at most once per repetition over the
    //! whole `K -> 1` sweep (that is what the tile-level novelty skip and the
    //! per-pair label test buy), but the *same* pair may well be found by
    //! several repetitions -- hence the `unique`. Duplicates cost nothing in
    //! correctness, since Kruskal rejects the second copy as a cycle, only
    //! work in the merge.
    //!
    //! `cutoff` is lowered afterwards, and only ever lowered: `local_tree` came
    //! out of a merge of a spanning tree with more edges, so it is the minimum
    //! spanning tree of a *superset* of the graph the old tree spanned, and the
    //! heaviest edge of an MST cannot go up when edges are added. The pruning
    //! is therefore monotone, and each tile is evaluated against a threshold no
    //! looser than the one the correctness argument above assumes.
    static void flush_buffer( std::vector<Edge>& buffer,
                              std::vector<Edge>& local_tree,
                              std::vector<Edge>& merged,
                              std::vector<Edge>& sort_scratch,
                              DSU& dsu,
                              float& cutoff,
                              size_t n,
                              double& seconds_spent ) {
        const auto start = std::chrono::steady_clock::now();
        sort_edges_by_weight( buffer, sort_scratch );
        /// `search_pairs` clears its tile buffer at the start of every call, so
        /// a buffer never spans two repetitions and, within one repetition, no
        /// pair is reported twice. This is therefore a cheap guard rather than
        /// a working filter -- and any duplicate it did miss would merely be
        /// rejected by the merge below as a cycle.
        buffer.erase( std::unique( buffer.begin(), buffer.end() ), buffer.end() );

        merged.clear();
        kruskal_merge( local_tree, buffer, dsu, merged );
        local_tree.swap( merged );
        buffer.clear();

        if ( local_tree.size() != n - 1 ) {
            // A spanning tree in must give a spanning tree out: the merge keeps
            // every vertex connected that was connected before.
            throw std::runtime_error(
                "pair_forest_emst: a spanning tree in must give a spanning tree out" );
        }

        // `std::min` only restates the argument above; it is never the binding
        // constraint, and it makes the monotonicity unconditional.
        cutoff = std::min( cutoff, cutoff_from( local_tree ) );

        seconds_spent +=
            std::chrono::duration<double>( std::chrono::steady_clock::now() - start ).count();
    }

    //! Carries an exception out of an OpenMP parallel region.
    //!
    //! OpenMP requires an exception thrown inside a region to be caught by the
    //! same thread inside that region; one that escapes is undefined behaviour,
    //! and with libgomp it is `std::terminate`. The realistic culprit here is
    //! `std::bad_alloc` from the per-thread buffers, which must surface as a
    //! catchable error, not an abort. `run` catches whatever its callable
    //! throws and keeps the first one; the caller rethrows it once the region
    //! has ended.
    //!
    //! `run` must be called *inside* the body of any worksharing loop, never
    //! around it: leaving a `#pragma omp for` early is itself non-conforming,
    //! and would leave the other threads waiting at its implicit barrier.
    class ParallelExceptionGuard {
    public:
        //! Calls `f`, returning `false` if it threw. Safe to call concurrently.
        template <typename F>
        bool run( F&& f ) noexcept {
            try {
                f();
                return true;
            } catch ( ... ) {
#pragma omp critical( pair_emst_exception_guard )
                if ( !first ) {
                    first = std::current_exception();
                }
                return false;
            }
        }

        //! Rethrows the first captured exception, if any. Call it after the
        //! region, from the thread that opened it.
        void rethrow_if_failed() const {
            if ( first ) {
                std::rethrow_exception( first );
            }
        }

    private:
        std::exception_ptr first;
    };

    //! Reduces the per-thread spanning trees to one, by pairwise halving.
    //!
    //! Every input is a spanning tree over the same vertex set, and
    //! `kruskal_merge` of two spanning trees is the minimum spanning tree of
    //! their union -- itself a spanning tree. The operation is associative and
    //! commutative on minimum spanning forests, so the halving order does not
    //! affect the result, only how much of the work runs in parallel.
    static std::vector<Edge> reduce_forests( std::vector<std::vector<Edge>>& forests, size_t n ) {
        if ( forests.empty() ) {
            return {};
        }

        size_t alive = forests.size();
        while ( alive > 1 ) {
            const size_t merges = alive / 2;
            ParallelExceptionGuard guard;
#pragma omp parallel for schedule( dynamic, 1 )
            for ( size_t i = 0; i < merges; i++ ) {
                guard.run( [&] {
                    /// A DSU per merge, private to the thread running it:
                    /// `kruskal_merge` resets the one it is handed, so two
                    /// concurrent merges must not be looking at the same one.
                    DSU dsu( n );
                    std::vector<Edge> merged;
                    kruskal_merge( forests[i], forests[alive - 1 - i], dsu, merged );
                    forests[i].swap( merged );
                } );
            }
            guard.rethrow_if_failed();
            /// Slot `i` absorbed slot `alive - 1 - i`, so the survivors are the
            /// first `alive - merges` slots. When `alive` is odd the middle slot
            /// took part in no merge and is already among them.
            alive -= merges;
        }
        return std::move( forests[0] );
    }

    //! Step 4: one batch of `width` repetitions at prefix `k`.
    //!
    //! `best` and `components` are read-only for the whole batch: nothing in
    //! here writes to shared state except each thread's own slot of `forests`
    //! and the distance counter, which is reduced.
    //!
    //! Returns the minimum spanning tree of `best` together with every edge the
    //! batch's repetitions turned up, and adds the batch's distance
    //! evaluations to `distances_computed`.
    template <typename Dataset, typename Hasher, typename Distance>
    static std::vector<Edge> run_batch( const PairForestIndex<Dataset, Hasher, Distance>& index,
                                        const std::vector<Edge>& best,
                                        const uint32_t* components,
                                        uint8_t k,
                                        size_t begin,
                                        size_t width,
                                        size_t buffer_edges,
                                        size_t& distances_computed ) {
        using ForestIndex = PairForestIndex<Dataset, Hasher, Distance>;

        const size_t n = index.num_points();
        const size_t num_threads = pair_emst_worker_count( width );

        /// One published tree per *thread*. A thread that is handed several
        /// repetitions folds all of them into the same tree, which is the whole
        /// reason the collector of `find_tree` is not needed here.
        std::vector<std::vector<Edge>> forests( num_threads );
        size_t batch_distances = 0;

        /// Thread-seconds, not wall-seconds: summed over the threads, so the
        /// two add up to roughly `num_threads` times the wall time of the
        /// region. What they are for is the *ratio* -- how much of a batch goes
        /// into looking at points and how much into folding the results in.
        double enumerate_seconds = 0.0;
        double flush_seconds = 0.0;
        ParallelExceptionGuard guard;

#pragma omp parallel num_threads( num_threads ) reduction( + : batch_distances ) \
    reduction( + : enumerate_seconds ) reduction( + : flush_seconds )
        {
            /// Per-thread state, declared inside the region so that it survives
            /// every repetition this thread picks up below: the tree it is
            /// building, its own cutoff, the index scratch (which caches the
            /// novelty labels of the last tree it looked at), and the merge
            /// buffers. None of it is shared, so none of it needs a lock.
            ///
            /// The declarations do not allocate; the allocations happen under
            /// `guard`, so a `std::bad_alloc` is carried out of the region
            /// instead of terminating the process.
            std::vector<Edge> local_tree;
            std::vector<Edge> merged;
            std::vector<Edge> sort_scratch;
            DSU dsu( 0 );
            typename ForestIndex::SearchScratch scratch;
            float cutoff = 0.0f;
            std::function<bool( std::vector<Edge>& )> flush;

            bool healthy = guard.run( [&] {
                /// Reserved up front at the sizes `buffer_edges_within_budget`
                /// accounted for. Left to grow on their own, the vectors
                /// would double past them and the memory cap would not hold.
                const size_t tile_edges =
                    static_cast<size_t>( PairCompactTree::TILE_SIZE ) * PairCompactTree::TILE_SIZE;
                scratch.tile_buffer.reserve( buffer_edges + tile_edges );
                sort_scratch.reserve( buffer_edges + tile_edges );
                merged.reserve( n - 1 );
                local_tree = best;
                dsu = DSU( static_cast<uint32_t>( n ) );
                cutoff = cutoff_from( local_tree );
                /// `search_pairs` holds a reference to `cutoff`, so lowering it
                /// in here prunes the very next tile. Built once rather than
                /// per repetition: wrapping a lambda in a `std::function`
                /// allocates.
                flush = [&]( std::vector<Edge>& buffer ) {
                    flush_buffer(
                        buffer, local_tree, merged, sort_scratch, dsu, cutoff, n, flush_seconds );
                    return false; // never stop early: the sweep decides when to stop
                };
            } );

            const auto thread_start = std::chrono::steady_clock::now();
#pragma omp for schedule( dynamic, 1 )
            for ( size_t rep = begin; rep < begin + width; rep++ ) {
                /// A thread that has failed still takes its share of the
                /// iterations, doing nothing with them: every thread of the
                /// team must reach the loop's implicit barrier.
                if ( !healthy ) {
                    continue;
                }
                healthy = guard.run( [&] {
                    batch_distances += index.search_pairs(
                        rep, k, cutoff, buffer_edges, flush, scratch, components );
                } );
            }
            enumerate_seconds +=
                std::chrono::duration<double>( std::chrono::steady_clock::now() - thread_start )
                    .count();

            if ( healthy ) {
                forests[omp_get_thread_num()] = std::move( local_tree );
            }
        }
        guard.rethrow_if_failed();
        // The flushes happen inside the loop that `enumerate_seconds` brackets.
        enumerate_seconds -= flush_seconds;

        const auto reduce_start = std::chrono::steady_clock::now();
        std::vector<Edge> reduced = reduce_forests( forests, n );
        const double reduce_seconds =
            std::chrono::duration<double>( std::chrono::steady_clock::now() - reduce_start )
                .count();

        // clang-format off
        LOG_INFO( "logger", "pair-emst",
                  "msg", "batch done",
                  "prefix", static_cast<size_t>( k ),
                  "repetitions", begin + width,
                  "threads", num_threads,
                  "distances", batch_distances,
                  "enumerate_thread_s", enumerate_seconds,
                  "flush_thread_s", flush_seconds,
                  "reduce_s", reduce_seconds );
        // clang-format on

        distances_computed += batch_distances;
        return reduced;
    }

    //! Step 4, continued: has the tree got close enough to optimal to stop?
    //!
    //! Edges lighter than the distance that the index has, by now, probed with
    //! failure probability at most `delta_per_pair` are *confirmed*: every pair
    //! that close has been looked at. Each unconfirmed edge is at least as
    //! heavy as the heaviest confirmed one, which turns the split into a lower
    //! bound on the true MST weight.
    template <typename Dataset, typename Hasher, typename Distance>
    static PairEmstStop check_stopping( const PairForestIndex<Dataset, Hasher, Distance>& index,
                                        const std::vector<Edge>& tree,
                                        float epsilon,
                                        float delta_per_pair,
                                        uint8_t k,
                                        size_t repetitions_done ) {
        const float confirmed_distance =
            index.distance_at_failure_probability( delta_per_pair, k, repetitions_done );
        const StoppingConditionInfo info = stopping_condition<Distance>( tree, confirmed_distance );

        const float weight_lower_bound =
            info.confirmed_weight + info.edges_to_confirm * info.heaviest_confirmed_edge;
        const bool should_stop = info.total_weight <= ( 1 + epsilon ) * weight_lower_bound;

        // The same fields the collector of `find_tree` logs, so that runs of the
        // two searches stay comparable line for line.
        // clang-format off
        LOG_INFO( "logger", "pair-emst",
                  "stop.total_weight", info.total_weight,
                  "stop.confirmed_weight", info.confirmed_weight,
                  "stop.heaviest_confirmed_edge", info.heaviest_confirmed_edge,
                  "stop.edges_to_confirm", info.edges_to_confirm,
                  "heaviest_edge", tree.back().weight,
                  "weight_lower_bound", weight_lower_bound,
                  "should_stop", should_stop );
        // clang-format on

        return PairEmstStop{
            .info = info, .weight_lower_bound = weight_lower_bound, .should_stop = should_stop };
    }

    //! Approximate Euclidean minimum spanning tree over `data`, to within a
    //! factor `1 + epsilon` with probability at least `1 - delta`.
    //!
    //! This overload takes the hash builder explicitly, for families whose
    //! builder needs more than the dimensionality of the data.
    //!
    //! Throws `std::invalid_argument` on fewer than two points, and
    //! `std::runtime_error` when the whole `K x repetitions` sweep is exhausted
    //! without the stopping rule ever firing.
    template <typename Dataset, typename Hasher, typename Distance>
    PairEmstResult pair_forest_emst( const Dataset& data,
                                     float epsilon,
                                     float delta,
                                     size_t repetitions,
                                     typename Hasher::Builder builder ) {
        Timer _t( "pair-forest-emst" );
        using ForestIndex = PairForestIndex<Dataset, Hasher, Distance>;

        const size_t n = data.size();
        if ( n < 2 ) {
            throw std::invalid_argument( "pair_forest_emst: needs at least two points" );
        }

        /// Union bound over the `n - 1` edges of a spanning tree: for the tree
        /// as a whole to be right with probability `1 - delta`, each edge must
        /// be right with probability `1 - delta / (n - 1)`. This is the same
        /// budget `EMST::find_tree` uses.
        const float delta_per_pair = delta / static_cast<float>( n - 1 );

        // --- 1. seed --------------------------------------------------------
        std::vector<Edge> best = seed_tree<Dataset, Distance>( data );

        // --- 2. index, built once -------------------------------------------
        const ForestIndex index = build_index<Dataset, Hasher, Distance>(
            data, std::move( builder ), repetitions, best, delta_per_pair );
        // clang-format off
        LOG_INFO( "msg", "pair forest index constructed",
                  "L", index.num_repetitions(),
                  "K", static_cast<size_t>( ForestIndex::K ),
                  "num_data", n,
                  "delta", delta,
                  "epsilon", epsilon,
                  "family", index.describe_family(),
                  "index_size_Gbytes", static_cast<double>( index.memory_usage() ) / ( 1 << 30 ) );
        // clang-format on

        // --- 3. running state ------------------------------------------------
        /// Sized after the index is built, so that the memory the index took
        /// is no longer counted as available. The thread count is the one a
        /// full batch runs with, the largest any batch uses.
        const size_t max_threads = pair_emst_worker_count( PAIR_EMST_BATCH_REPETITIONS );
        const size_t memory_budget = static_cast<size_t>(
            PAIR_EMST_MEMORY_FRACTION * static_cast<double>( available_memory_bytes() ) );
        const size_t buffer_edges = buffer_edges_within_budget( n, max_threads, memory_budget );
        // clang-format off
        LOG_INFO( "msg", "per-thread edge buffers sized",
                  "threads", max_threads,
                  "buffer_edges", buffer_edges,
                  "buffer_edges_per_point", static_cast<double>( buffer_edges ) / n,
                  "memory_budget_Gbytes", static_cast<double>( memory_budget ) / ( 1 << 30 ) );
        // clang-format on

        /// Edges of `best` that the stopping rule has already confirmed. Both
        /// endpoints of a confirmed edge are permanently in the same component
        /// of the final tree, so a pair inside one component can be skipped
        /// outright -- it can only ever close a cycle.
        DSU confirmed( static_cast<uint32_t>( n ) );
        size_t confirmed_edges = 0;
        std::vector<uint32_t> components( n );
        size_t distances_computed = 0;

        // --- 4. sweep prefixes from the longest to the shortest ---------------
        for ( uint8_t k = ForestIndex::K; k >= 1; k-- ) {
            for ( size_t begin = 0; begin < repetitions; begin += PAIR_EMST_BATCH_REPETITIONS ) {
                const size_t width = std::min( PAIR_EMST_BATCH_REPETITIONS, repetitions - begin );

                /// Materialised once per batch so that the inner kernel reads a
                /// flat array rather than chasing a DSU (or, worse, calling
                /// through a `std::function` as `Index::search_pairs_different_
                /// groups` does). Frozen for the batch, like everything else.
                confirmed.compress_all();
                for ( size_t i = 0; i < n; i++ ) {
                    components[i] = confirmed.get_parent( static_cast<uint32_t>( i ) );
                }
                /// With nothing confirmed yet every point is its own
                /// component and the test can never fire, so passing no array
                /// at all makes `dispatch_tile` pick the kernel that does not
                /// carry the test in its inner loop.
                const uint32_t* component_filter =
                    ( confirmed_edges > 0 ) ? components.data() : nullptr;

                std::vector<Edge> batch_tree =
                    run_batch<Dataset, Hasher, Distance>( index,
                                                          best,
                                                          component_filter,
                                                          k,
                                                          begin,
                                                          width,
                                                          buffer_edges,
                                                          distances_computed );

                /// `batch_tree` already contains `best` (every thread started
                /// from it), so this merge is cheap; it is here so that the
                /// invariant "`best` is the MST of everything seen so far" does
                /// not depend on that fact.
                {
                    DSU dsu( static_cast<uint32_t>( n ) );
                    std::vector<Edge> merged;
                    kruskal_merge( best, batch_tree, dsu, merged );
                    best.swap( merged );
                }

                const size_t repetitions_done = begin + width;
                const PairEmstStop stop = check_stopping<Dataset, Hasher, Distance>(
                    index, best, epsilon, delta_per_pair, k, repetitions_done );

                if ( stop.should_stop ) {
                    // clang-format off
                    LOG_INFO( "msg", "tree found",
                              "prefix", static_cast<size_t>( k ),
                              "repetitions", repetitions_done,
                              "distances_computed", distances_computed );
                    // clang-format on
                    /// Everything above needs only weight order, which is all
                    /// the radix sort in `flush_buffer` provides. Callers are
                    /// promised the full `Edge::operator<` order, so it is
                    /// established once, here, over `n - 1` edges.
                    std::sort( best.begin(), best.end() );
                    return PairEmstResult{ .tree = std::move( best ),
                                           .weight = stop.info.total_weight,
                                           .distances_computed = distances_computed,
                                           .prefix_at_stop = static_cast<size_t>( k ),
                                           .repetitions_at_stop = repetitions_done,
                                           .index_bytes = index.memory_usage() };
                }

                /// Refresh the confirmed components from the confirmed prefix of
                /// the new tree. The prefix is a prefix of a *sorted* tree, and
                /// the confirmed distance only grows as the sweep goes on, so
                /// the component structure only ever coarsens.
                confirmed.reset();
                for ( size_t idx = 0; idx < stop.info.confirmed_edges; idx++ ) {
                    const Edge& e = best.at( idx );
                    confirmed.union_sets( e.a, e.b );
                }
                confirmed.compress_all();
                confirmed_edges = stop.info.confirmed_edges;
            }
        }

        throw std::runtime_error( "Minimum spanning tree not found" );
    }

    //! Approximate Euclidean minimum spanning tree over `data`, building the
    //! hash builder from the dimensionality of the data. See the overload above.
    template <typename Dataset, typename Hasher, typename Distance>
    PairEmstResult
    pair_forest_emst( const Dataset& data, float epsilon, float delta, size_t repetitions ) {
        return pair_forest_emst<Dataset, Hasher, Distance>(
            data, epsilon, delta, repetitions, typename Hasher::Builder( data.get_dimensions() ) );
    }

} // namespace panna
