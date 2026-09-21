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
#include <atomic>
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
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "panna/core_distances.hpp"
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
    //!
    //! Templated on the edge type so that the mutual-reachability search can
    //! use it on its `MREdge` trees, whose `weight` is the mutual-reachability
    //! weight; see `flush_buffer_mr` for why that bounds a *raw* distance.
    template <typename E>
    static float cutoff_from( const std::vector<E>& tree ) {
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

    //! The arithmetic behind `buffer_edges_within_budget` and
    //! `mr_buffer_edges_within_budget`: how many edges each of `threads`
    //! threads may buffer when `shared_bytes` are taken off the top of
    //! `budget_bytes`, and each thread then needs `fixed_bytes` plus
    //! `bytes_per_buffered_edge` for every buffered edge.
    //!
    //! A buffer overshoots its nominal size by up to one tile
    //! (`TILE_SIZE^2` edges), since `search_pairs` checks the size only after
    //! appending a whole tile, so that overshoot is paid for up front. Whatever
    //! is left goes to the buffers, up to `PAIR_EMST_BUFFER_EDGES_PER_POINT * n`
    //! edges.
    //!
    //! Throws `std::runtime_error`, naming `who`, when not even
    //! `PAIR_EMST_MIN_BUFFER_EDGES_PER_POINT * n` edges fit.
    static size_t buffer_edges_for_costs( const char* who,
                                          size_t n,
                                          size_t threads,
                                          size_t budget_bytes,
                                          size_t shared_bytes,
                                          size_t fixed_bytes,
                                          size_t bytes_per_buffered_edge ) {
        const size_t tile_edges =
            static_cast<size_t>( PairCompactTree::TILE_SIZE ) * PairCompactTree::TILE_SIZE;

        const size_t wanted = PAIR_EMST_BUFFER_EDGES_PER_POINT * n;
        const size_t minimum = PAIR_EMST_MIN_BUFFER_EDGES_PER_POINT * n;

        const size_t thread_budget =
            ( budget_bytes > shared_bytes ) ? budget_bytes - shared_bytes : 0;
        const size_t per_thread_bytes = thread_budget / std::max<size_t>( 1, threads );
        const size_t buffer_bytes =
            ( per_thread_bytes > fixed_bytes ) ? per_thread_bytes - fixed_bytes : 0;
        const size_t affordable_with_overshoot = buffer_bytes / bytes_per_buffered_edge;
        const size_t affordable =
            ( affordable_with_overshoot > tile_edges ) ? affordable_with_overshoot - tile_edges : 0;

        if ( affordable < minimum ) {
            throw std::runtime_error(
                std::string( who ) + ": not enough memory for the per-thread edge buffers: " +
                std::to_string( threads ) + " threads need at least " +
                std::to_string( shared_bytes +
                                threads * ( fixed_bytes +
                                            bytes_per_buffered_edge * ( minimum + tile_edges ) ) ) +
                " bytes, but the memory budget is only " + std::to_string( budget_bytes ) +
                " bytes" );
        }
        return std::min( wanted, affordable );
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
        return buffer_edges_for_costs( "pair_forest_emst",
                                       n,
                                       threads,
                                       budget_bytes,
                                       0,
                                       n * ( 2 * sizeof( Edge ) + 3 * sizeof( uint32_t ) ),
                                       2 * sizeof( Edge ) );
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
    //!
    //! Templated on the record type, which only needs a `float weight` member:
    //! the mutual-reachability search sorts `MREdge`s with it, keyed on their
    //! mutual-reachability weight.
    template <typename E>
    static void sort_edges_by_weight( std::vector<E>& edges, std::vector<E>& scratch ) {
        const size_t m = edges.size();
        if ( m < 2 ) {
            return;
        }

        auto key = []( const E& e ) -> uint32_t {
            uint32_t bits = 0;
            std::memcpy( &bits, &e.weight, sizeof( bits ) );
            const uint32_t sign_mask = ( bits & 0x80000000u ) ? 0xFFFFFFFFu : 0x80000000u;
            return bits ^ sign_mask;
        };

        /// All four histograms are built in a single pass over the data, so the
        /// records are read five times in total rather than eight.
        uint32_t counts[4][256] = {};
        for ( const E& e : edges ) {
            const uint32_t k = key( e );
            counts[0][k & 0xff]++;
            counts[1][( k >> 8 ) & 0xff]++;
            counts[2][( k >> 16 ) & 0xff]++;
            counts[3][( k >> 24 ) & 0xff]++;
        }

        scratch.resize( m );
        std::vector<E>* src = &edges;
        std::vector<E>* dst = &scratch;

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
            for ( const E& e : *src ) {
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
    //!
    //! Templated on the edge type, for the `MREdge` trees of the
    //! mutual-reachability search.
    template <typename E>
    static std::vector<E> reduce_forests( std::vector<std::vector<E>>& forests, size_t n ) {
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
                    std::vector<E> merged;
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

    // =======================================================================
    // Mutual-reachability distance
    // =======================================================================
    //
    //! HDBSCAN wants the minimum spanning tree under the *mutual-reachability*
    //! distance `mr(a, b) = max(d(a, b), core(a), core(b))`, where `core(p)` is
    //! the distance from `p` to its `num_neighbors`-th nearest neighbor. The
    //! search below is `pair_forest_emst` with the core distances estimated
    //! on the fly, from the very pairs the sweep turns up.
    //!
    //! **The difficulty.** Core estimates only improve -- `CoreDistances` only
    //! inserts real pairs at their real distances -- so they only go *down*,
    //! and when `core(p)` drops the weight of every edge at `p` drops with it.
    //! A tree kept sorted by the weights of some past moment is therefore
    //! stale, and an edge rejected by an old Kruskal run may well be wanted
    //! now. `EMST::find_tree_mutual_reachability_distance` copes by storing
    //! the tree under the weights of the moment and keeping *every* non-tree
    //! edge around for later, which is exactly what makes its memory grow
    //! without bound. Here instead:
    //!
    //!  - a tree edge is an `MREdge`: it carries its **raw** distance in
    //!    `lower_bound`, and its mutual-reachability weight in `weight` only as
    //!    of the last time it was read. Weights are re-read from the core
    //!    distances right before each Kruskal run;
    //!  - a weight read at any moment is an **upper bound** on every later
    //!    weight of the same edge, and the raw distance is a **lower bound** on
    //!    all of them.
    //!
    //! **Invariant (N).** Every pair the sweep discovers is inserted into the
    //! core distances *before* its first Kruskal run. From then on, whenever
    //! `d(a, b) < core(a)`, `b` is in the neighborhood `NN(a)`: it went in at
    //! insertion, and the only way out is being evicted as the heap maximum,
    //! after which `core(a) <= d(a, b)` for good. An evicted pair is handed to
    //! the next Kruskal run of the thread that evicted it.
    //!
    //! **The retention rule.** An edge `e` a Kruskal run *rejects* closes a
    //! cycle of edges no heavier than `w(e)`. Then either
    //!
    //!  1. `e` is **tight**, `w(e) == d(e)`: it can be dropped for good. The
    //!     cycle's edges can only get lighter, and `e` can never weigh less than
    //!     `d(e)`, so `e` stays a maximum of that cycle forever (cycle
    //!     property); or
    //!  2. `e` is **not tight**, `d(e) < core(a)` or `d(e) < core(b)`: by (N) it
    //!     is stored in a neighborhood, which *is* the storage for such edges,
    //!     and the explicit copy can be dropped.
    //!
    //! So nothing needs storing beside the core distances (`n * num_neighbors`
    //! entries) but the spanning trees and the edges buffered for the next
    //! flush. The neighborhoods are re-examined once per batch, in
    //! `absorb_neighborhoods`, which is where a stored edge whose weight has
    //! dropped gets its chance to enter the tree. Only the edges at a point
    //! whose core dropped since the previous re-examination need it: any other
    //! stored edge weighs what it weighed then, when it was either kept or
    //! rejected with a cycle of no heavier edges as its certificate.
    //!
    //! **Sharing.** The core distances are one `SharedCoreDistances`, *live*
    //! and shared by every thread rather than frozen per batch: a frozen copy
    //! would force each thread to buffer all its non-tight edges until the
    //! batch ends (or to own an `n * num_neighbors` copy), which is the memory
    //! problem again. Case 1 of the rule is content with any weight read, since
    //! every read is an upper bound. Case 2 is not: dropping `e` because
    //! `core(a)` *reads* above `d(e)` is only safe if `e` really is in `NN(a)`
    //! -- or still on its way there -- which a stale read does not promise.
    //! Two threads evicting `e` from `NN(a)` and from `NN(b)` at once, each
    //! reading the other endpoint's pre-eviction core, would each file `e`
    //! under the other's neighborhood and lose it. The mirror of the cores is
    //! therefore accessed with sequentially consistent loads and stores, which
    //! makes every run of the search equivalent, as far as the cores are
    //! concerned, to *some* interleaving of the threads' steps, and in any
    //! interleaving the rule holds: a thread that finds `core(a) > d(e)`
    //! finds `e` in `NN(a)`, or finds it evicted by someone who will process
    //! it afterwards, or finds the insertion of `e` still under way, in which
    //! case the discovering thread processes `e` afterwards. (A fence between
    //! the insertions and the reads of a flush would settle the two-eviction
    //! race, but not a chain through a third thread's insertion.)

    //! What `pair_forest_emst_mutual_reachability` returns.
    struct PairMrEmstResult {
        //! Sorted ascending, `n - 1` edges. Each weight is the Euclidean
        //! mutual-reachability distance, `max(to_euclidean(d),
        //! to_euclidean(core(a)), to_euclidean(core(b)))`, exactly as
        //! `EMST::find_tree_mutual_reachability_distance` reports it.
        std::vector<Edge> tree;
        //! The final core-distance estimates, in raw `Distance` units.
        CoreDistances core_distances;
        float weight;               //!< sum of the weights of `tree`
        size_t distances_computed;  //!< distance evaluations of the seeding and of the sweep
        size_t prefix_at_stop;      //!< the prefix length the stopping rule fired at
        size_t repetitions_at_stop; //!< repetitions probed at that prefix
        size_t index_bytes;         //!< as in `PairEmstResult`
    };

    //! Observation points and switches of `pair_forest_emst_mutual_
    //! reachability`, for tests. Everything defaults to "do nothing different",
    //! and a caller that is not a test has no reason to pass one.
    struct PairMrEmstHooks {
        //! Whether to seed the core distances from the index (step 3). Turning
        //! it off leaves the cores poor when the sweep starts, which is what a
        //! test of the retention rule wants: it makes them drop a lot later.
        bool seed_from_index = true;
        //! Called once, before the initial tree is built, with the seed tree
        //! and the core distances the sweep will start from.
        std::function<void( const std::vector<Edge>&, const CoreDistances& )> on_start;
        //! Called with every buffer of raw pairs handed to a flush, before the
        //! flush looks at it. Called **concurrently**, from the worker threads.
        std::function<void( const std::vector<Edge>& )> on_flush;
        //! Called after every batch, with the tree (weighted by the current
        //! mutual-reachability weights, sorted) and the core distances, both
        //! quiescent.
        std::function<void( const std::vector<MREdge>&, const CoreDistances& )> on_batch;
    };

    //! One `CoreDistances`, shared by every thread of the search.
    //!
    //! Writes go through `insert`, which takes a per-point `std::atomic_flag`
    //! spinlock (as `CoreDistances::refine` does) around each side of the
    //! pair. Reads never touch the heaps: every point's current core distance
    //! is mirrored in a `std::atomic<float>`, stored under the lock after each
    //! heap change, so reading a weight costs two 4-byte loads and never races
    //! with a writer.
    //!
    //! The mirror is loaded and stored `seq_cst`, for the reason given in the
    //! section comment above. On x86 a `seq_cst` load is a plain `mov`, so
    //! the reads, which are the frequent operation, cost nothing extra; the
    //! stores, which happen only when a core actually drops, are an `xchg`
    //! inside a critical section that has already paid for a locked
    //! instruction.
    //!
    //! Each point also carries a *lowered* flag, raised whenever its core
    //! drops and cleared by `clear_lowered`: `absorb_neighborhoods` uses it to
    //! re-examine only the stored edges whose weight can have changed.
    //!
    //! With `num_neighbors == 0` there are no core distances: the mirror holds
    //! `-inf`, every insertion is a no-op, and `weight` is the raw distance.
    //! (`pair_forest_emst_mutual_reachability` does not get here with zero
    //! neighbors, it runs the plain search; the class is correct regardless.)
    class SharedCoreDistances {
    public:
        explicit SharedCoreDistances( CoreDistances&& estimates ):
            estimates( std::move( estimates ) ),
            locks( this->estimates.size() ),
            mirror( this->estimates.size() ),
            lowered( this->estimates.size() ) {
            sync_mirror();
            clear_lowered();
        }

        SharedCoreDistances( const SharedCoreDistances& ) = delete;
        SharedCoreDistances& operator=( const SharedCoreDistances& ) = delete;

        size_t size() const {
            return estimates.size();
        }

        bool has_cores() const {
            return estimates.get_num_neighbors() > 0;
        }

        //! The underlying estimates, for a caller that knows no other thread is
        //! using them -- `refine`, and the scans at the end of a batch. After
        //! modifying them, call `sync_mirror`.
        CoreDistances& exclusive() {
            return estimates;
        }

        const CoreDistances& exclusive() const {
            return estimates;
        }

        //! Rebuilds the mirror from the heaps. Not thread-safe.
        void sync_mirror() {
            const bool cores = has_cores();
            for ( size_t i = 0; i < estimates.size(); i++ ) {
                mirror[i].store( cores ? estimates.core_distance( static_cast<uint32_t>( i ) )
                                       : -std::numeric_limits<float>::infinity() );
            }
        }

        //! The current estimate of `core(a)`; see the class comment.
        float core( uint32_t a ) const {
            return mirror[a].load( std::memory_order_seq_cst );
        }

        //! The largest current core estimate: no pair at least this far apart
        //! can enter any neighborhood. Not thread-safe.
        float max_core() const {
            float out = -std::numeric_limits<float>::infinity();
            for ( size_t i = 0; i < mirror.size(); i++ ) {
                out = std::max( out, mirror[i].load( std::memory_order_relaxed ) );
            }
            return out;
        }

        //! The mutual-reachability weight of the pair `(a, b)` at raw distance
        //! `d`, under the current estimates: an upper bound on its true value,
        //! and on every weight read later.
        float weight( uint32_t a, uint32_t b, float d ) const {
            return std::max( d, std::max( core( a ), core( b ) ) );
        }

        //! Whether `core(a)` dropped since the last `clear_lowered`. Meant for
        //! when no thread is inserting.
        bool is_lowered( uint32_t a ) const {
            return lowered[a].load( std::memory_order_relaxed ) != 0;
        }

        //! Whether any core dropped since the last `clear_lowered`. Not
        //! thread-safe.
        bool any_lowered() const {
            for ( size_t i = 0; i < lowered.size(); i++ ) {
                if ( is_lowered( static_cast<uint32_t>( i ) ) ) {
                    return true;
                }
            }
            return false;
        }

        //! Not thread-safe.
        void clear_lowered() {
            for ( auto& flag : lowered ) {
                flag.store( 0, std::memory_order_relaxed );
            }
        }

        //! Inserts the pair `(a, b)` at distance `d` into both neighborhoods,
        //! appending to `evicted` every real neighbor this pushed out, as a raw
        //! `Edge` from the point whose neighborhood lost it. Thread-safe.
        void insert( uint32_t a, uint32_t b, float d, std::vector<Edge>& evicted ) {
            insert_one_sided( a, b, d, evicted );
            insert_one_sided( b, a, d, evicted );
        }

        //! Makes (N) hold for every pair already stored: afterwards, whenever
        //! `b` is in `NN(a)` at distance `d` and `d < core(b)`, `a` is in
        //! `NN(b)` too. Every other insertion of the search goes both ways,
        //! but `CoreDistances::random` fills each neighborhood one-sidedly,
        //! and a pair stored on one side only would be lost the moment it was
        //! evicted from that side and then rejected by a Kruskal run: its
        //! other endpoint's core, still above `d`, would pass it off as stored
        //! there.
        //!
        //! One pass over a snapshot of the pairs, inserting each into the
        //! other endpoint's neighborhood, is enough. After the pass over
        //! `(a, b)`, either `a` is in `NN(b)` or `core(b) <= d`, and both stay
        //! true until `a` is evicted from `NN(b)`, which leaves
        //! `core(b) <= d`. A pair pushed out along the way disappears from
        //! both sides at once, never to be seen by a Kruskal run; like the
        //! pairs looked at by the seeding, it is a pair the search is not
        //! bound to keep. Runs in parallel, but no other thread may be
        //! inserting.
        void symmetrize() {
            if ( !has_cores() ) {
                return;
            }
            const size_t k = estimates.get_num_neighbors();
            const std::vector<std::pair<float, uint32_t>> snapshot = estimates.all();
            const size_t n = estimates.size();
            constexpr uint32_t empty = std::numeric_limits<uint32_t>::max();
#pragma omp parallel
            {
                std::vector<Edge> dropped;
#pragma omp for schedule( dynamic, 1024 )
                for ( size_t a = 0; a < n; a++ ) {
                    for ( size_t j = 0; j < k; j++ ) {
                        const auto& [d, b] = snapshot[a * k + j];
                        if ( b != empty ) {
                            insert_one_sided( b, static_cast<uint32_t>( a ), d, dropped );
                            dropped.clear();
                        }
                    }
                }
            }
        }

        //! Hands the estimates over. The object must not be used afterwards.
        CoreDistances release() {
            return std::move( estimates );
        }

    private:
        CoreDistances estimates;
        std::vector<std::atomic_flag> locks;
        std::vector<std::atomic<float>> mirror;
        //! One byte per point rather than a bit, so that raising a flag is a
        //! plain store to an object no other point shares.
        std::vector<std::atomic<uint8_t>> lowered;

        void insert_one_sided( uint32_t src, uint32_t dst, float d, std::vector<Edge>& evicted ) {
            /// `CoreDistances::do_update` accepts only `d < core(src)`, and the
            /// mirror never reads below the true core, so a pair failing this
            /// test would be turned away under the lock anyway. It is by far
            /// the common case once the estimates settle, and it takes no lock
            /// and writes no shared cache line. (It also turns away NaNs.)
            if ( !( d < core( src ) ) ) {
                return;
            }
            std::optional<std::pair<float, uint32_t>> out;
            while ( locks[src].test_and_set( std::memory_order_acquire ) ) {
                // spin: the critical section is a scan of num_neighbors entries
            }
            const float before = estimates.core_distance( src );
            if ( estimates.update_one_sided_evicting( src, dst, d, out ) ) {
                const float after = estimates.core_distance( src );
                if ( after < before ) {
                    mirror[src].store( after, std::memory_order_seq_cst );
                    lowered[src].store( 1, std::memory_order_relaxed );
                }
            }
            locks[src].clear( std::memory_order_release );
            if ( out ) {
                evicted.push_back( Edge{ .weight = out->first, .a = src, .b = out->second } );
            }
        }
    };

    //! The mutual-reachability counterpart of `buffer_edges_within_budget`.
    //!
    //! Taken off the top of the budget, because there is one of each for the
    //! whole search:
    //!
    //!  - the core distances: `num_neighbors` `(float, uint32_t)` entries per
    //!    point, plus the 4-byte mirror and the 1-byte lock of
    //!    `SharedCoreDistances`;
    //!  - the end of a batch: `absorb_neighborhoods` gathers up to one
    //!    candidate per stored neighbor, as `MREdge`s, and radix-sorts them
    //!    through a scratch copy. That is the worst case, when every core
    //!    dropped during the batch; it is charged in full all the same.
    //!
    //! Per thread, what `run_batch_mr` reserves:
    //!
    //!  - `n` edges' worth, as `MREdge`, of each of `local_tree`, `merged`,
    //!    the candidates and their sort scratch: every tree edge whose weight
    //!    changed since the last flush becomes a candidate;
    //!  - the `DSU` and the novelty labels, as in the plain search;
    //!  - per buffered edge: the edge itself in the tile buffer, up to two
    //!    evicted neighbors (one per side of the insertion), and for each of
    //!    those three a candidate and its sort scratch.
    //!
    //! Evictions that many are the worst case, not the common one; they are
    //! reserved for all the same, because the buffers are reserved up front and
    //! a buffer that outgrew its reservation would double past the budget.
    static size_t mr_buffer_edges_within_budget( size_t n,
                                                 size_t num_neighbors,
                                                 size_t threads,
                                                 size_t budget_bytes ) {
        const size_t core_bytes = n * ( num_neighbors * sizeof( std::pair<float, uint32_t> ) +
                                        sizeof( std::atomic<float> ) + sizeof( std::atomic_flag ) );
        const size_t batch_end_bytes = 2 * n * num_neighbors * sizeof( MREdge );
        const size_t shared_bytes = core_bytes + batch_end_bytes;
        if ( shared_bytes > budget_bytes ) {
            throw std::runtime_error(
                "pair_forest_emst_mutual_reachability: not enough memory for the shared "
                "core-distance state: " +
                std::to_string( num_neighbors ) + " neighbors per point need " +
                std::to_string( shared_bytes ) +
                " bytes before any per-thread buffer, but the "
                "memory budget is only " +
                std::to_string( budget_bytes ) + " bytes" );
        }
        return buffer_edges_for_costs( "pair_forest_emst_mutual_reachability",
                                       n,
                                       threads,
                                       budget_bytes,
                                       shared_bytes,
                                       n * ( 4 * sizeof( MREdge ) + 3 * sizeof( uint32_t ) ),
                                       3 * sizeof( Edge ) + 3 * 2 * sizeof( MREdge ) );
    }

    //! Sets every weight of `tree` to its current mutual-reachability weight,
    //! then sorts by it. `scratch` is the radix sort's ping-pong buffer.
    static void reweight_and_sort( std::vector<MREdge>& tree,
                                   const SharedCoreDistances& shared,
                                   std::vector<MREdge>& scratch ) {
        for ( MREdge& e : tree ) {
            e.weight = shared.weight( e.a, e.b, e.lower_bound );
        }
        sort_edges_by_weight( tree, scratch );
    }

    //! Appends to `out` every pair stored in a neighborhood whose current
    //! mutual-reachability weight is at most `max_weight` -- or, with
    //! `only_lowered`, just those of them with an endpoint whose core dropped
    //! since the flags were last cleared. A pair stored in both neighborhoods
    //! is appended once, from the neighborhood of its smaller endpoint.
    //!
    //! Reads the heaps directly, so no thread may be writing them. Two passes
    //! over fixed chunks of points, one counting and one writing, let the scan
    //! run in parallel and write straight into `out` without per-thread copies.
    static void collect_neighborhood_edges( const SharedCoreDistances& shared,
                                            float max_weight,
                                            bool only_lowered,
                                            std::vector<MREdge>& out ) {
        const size_t n = shared.size();
        const size_t k = shared.exclusive().get_num_neighbors();
        if ( k == 0 || n == 0 ) {
            return;
        }
        const auto& neighbors = shared.exclusive().all();
        constexpr uint32_t empty = std::numeric_limits<uint32_t>::max();
        constexpr size_t chunk = 4096;
        const size_t num_chunks = ( n + chunk - 1 ) / chunk;

        auto stores = [&]( uint32_t p, uint32_t q ) {
            for ( size_t j = 0; j < k; j++ ) {
                if ( neighbors[static_cast<size_t>( p ) * k + j].second == q ) {
                    return true;
                }
            }
            return false;
        };

        /// `emit( a, j )` is the candidate for the `j`-th neighbor slot of `a`,
        /// or `std::nullopt` if the slot is empty, the pair unaffected by the
        /// lowered cores, too heavy, or left for `b`'s neighborhood to emit.
        auto emit = [&]( size_t a, size_t j ) -> std::optional<MREdge> {
            const auto& [d, b] = neighbors[a * k + j];
            const uint32_t a32 = static_cast<uint32_t>( a );
            if ( b == empty ) {
                return std::nullopt;
            }
            if ( only_lowered && !shared.is_lowered( a32 ) && !shared.is_lowered( b ) ) {
                return std::nullopt;
            }
            const float w = shared.weight( a32, b, d );
            if ( !( w <= max_weight ) ) {
                return std::nullopt;
            }
            if ( a32 > b && stores( b, a32 ) ) {
                return std::nullopt;
            }
            return MREdge{ .weight = w, .lower_bound = d, .a = a32, .b = b };
        };

        std::vector<size_t> offsets( num_chunks + 1, 0 );
#pragma omp parallel for schedule( dynamic, 1 )
        for ( size_t c = 0; c < num_chunks; c++ ) {
            size_t count = 0;
            for ( size_t a = c * chunk; a < std::min( n, ( c + 1 ) * chunk ); a++ ) {
                for ( size_t j = 0; j < k; j++ ) {
                    count += emit( a, j ).has_value();
                }
            }
            offsets[c + 1] = count;
        }
        const size_t base = out.size();
        for ( size_t c = 0; c < num_chunks; c++ ) {
            offsets[c + 1] += offsets[c];
        }
        out.resize( base + offsets[num_chunks] );
#pragma omp parallel for schedule( dynamic, 1 )
        for ( size_t c = 0; c < num_chunks; c++ ) {
            size_t pos = base + offsets[c];
            for ( size_t a = c * chunk; a < std::min( n, ( c + 1 ) * chunk ); a++ ) {
                for ( size_t j = 0; j < k; j++ ) {
                    if ( const auto e = emit( a, j ) ) {
                        out[pos++] = *e;
                    }
                }
            }
        }
    }

    //! The Kruskal run that re-examines the neighborhoods: the minimum spanning
    //! tree of `tree` and the stored neighbor pairs, all under the current
    //! weights. Clears the lowered flags.
    //!
    //! `tree` must be a spanning tree, reweighted and sorted (`reweight_and_
    //! sort`). Anything heavier than its heaviest edge closes a cycle of
    //! lighter edges and is rejected for sure, so it is not even gathered.
    //!
    //! This pass is what makes the result the minimum spanning tree of *every*
    //! retained edge: the flushes only look at the tree and at freshly found
    //! edges, never at a stored neighbor pair whose weight dropped since it was
    //! rejected. With `only_lowered` it looks only at the stored pairs with an
    //! endpoint whose core dropped since the previous pass, and at none at all
    //! if no core did: every other stored pair weighs what it did at that pass,
    //! which kept it or rejected it on a cycle of edges no heavier than it --
    //! edges that since then have only got lighter, or been dropped on a
    //! certificate of their own. The very first pass must look at everything.
    //!
    //! Reads the heaps, so the cores must be quiescent.
    static std::vector<MREdge> absorb_neighborhoods( std::vector<MREdge> tree,
                                                     SharedCoreDistances& shared,
                                                     bool only_lowered,
                                                     size_t n ) {
        if ( tree.size() != n - 1 ) {
            throw std::runtime_error( "pair_forest_emst_mutual_reachability: absorb_neighborhoods "
                                      "needs a spanning tree" );
        }
        if ( only_lowered && !shared.any_lowered() ) {
            return tree;
        }
        const float max_weight = tree.back().weight;

        std::vector<MREdge> candidates;
        collect_neighborhood_edges( shared, max_weight, only_lowered, candidates );
        shared.clear_lowered();

        std::vector<MREdge> scratch;
        sort_edges_by_weight( candidates, scratch );
        scratch = std::vector<MREdge>();

        DSU dsu( static_cast<uint32_t>( n ) );
        std::vector<MREdge> merged;
        merged.reserve( n - 1 );
        kruskal_merge( tree, candidates, dsu, merged );
        if ( merged.size() != n - 1 ) {
            throw std::runtime_error( "pair_forest_emst_mutual_reachability: a spanning tree in "
                                      "must give a spanning tree out" );
        }
        return merged;
    }

    //! The mutual-reachability counterpart of `flush_buffer`: folds one buffer
    //! of freshly enumerated raw edges into `local_tree`, an `MREdge` tree
    //! sorted by the weights it was last read under.
    //!
    //!  1. **(N).** Every buffered pair goes into the shared core distances
    //!     first, and the neighbors that pushes out are collected in `evicted`:
    //!     this flush is the "next Kruskal run" they are routed to.
    //!  2. **Reweight the tree.** An edge whose weight did not change stays
    //!     where it is, in a *stable run* that is still sorted; one whose weight
    //!     dropped moves to the candidates. `max_weight` is the heaviest edge of
    //!     the reweighted tree.
    //!  3. **Pre-filter.** Buffered and evicted edges are weighed and kept only
    //!     if no heavier than `max_weight`: anything heavier closes a cycle
    //!     through the reweighted tree, so the run below would reject it anyway,
    //!     and by the retention rule a rejected edge may be dropped.
    //!  4. **Merge.** The candidates are radix-sorted and merged with the stable
    //!     run. Rejected edges are simply not copied out, again by the rule.
    //!
    //! Steps 1-3 read the cores at slightly different moments while other
    //! threads keep writing them. Every read is an upper bound, which is all
    //! case 1 of the rule needs; case 2 also needs the reads to agree with
    //! where each edge is stored, which is what the `seq_cst` mirror of
    //! `SharedCoreDistances` provides (see the section comment). Unlike
    //! `flush_buffer` there is no `unique`: a duplicate is only rejected as a
    //! cycle, and duplicates are rare here too.
    //!
    //! The cutoff. `search_pairs` compares *raw* distances against it, and it
    //! is the heaviest mutual-reachability weight of the tree. A pair with
    //! `d > max_weight` is safe to discard without even inserting it: every
    //! point has a tree edge, whose weight bounds its core, so `d` exceeds
    //! both cores and the pair's weight is `d` forever; and the tree joins its
    //! endpoints through lighter edges. It is a rejected tight edge before it
    //! was ever computed. As in `flush_buffer`, the cutoff only goes down.
    static void flush_buffer_mr( std::vector<Edge>& buffer,
                                 SharedCoreDistances& shared,
                                 std::vector<MREdge>& local_tree,
                                 std::vector<MREdge>& merged,
                                 std::vector<MREdge>& candidates,
                                 std::vector<MREdge>& sort_scratch,
                                 std::vector<Edge>& evicted,
                                 DSU& dsu,
                                 float& cutoff,
                                 size_t n,
                                 const PairMrEmstHooks& hooks,
                                 double& seconds_spent ) {
        const auto start = std::chrono::steady_clock::now();
        if ( hooks.on_flush ) {
            hooks.on_flush( buffer );
        }

        // --- 1. insert first, so that (N) holds by the time of the merge ------
        evicted.clear();
        for ( const Edge& e : buffer ) {
            shared.insert( e.a, e.b, e.weight, evicted );
        }

        // --- 2. reweight the tree, splitting off the stable run ---------------
        candidates.clear();
        size_t stable = 0;
        float max_weight = -std::numeric_limits<float>::infinity();
        for ( size_t i = 0; i < local_tree.size(); i++ ) {
            MREdge e = local_tree[i];
            const float w = shared.weight( e.a, e.b, e.lower_bound );
            max_weight = std::max( max_weight, w );
            if ( w == e.weight ) {
                local_tree[stable++] = e;
            } else {
                e.weight = w;
                candidates.push_back( e );
            }
        }
        local_tree.resize( stable );

        // --- 3. weigh the new edges, dropping those that cannot enter ---------
        auto consider = [&]( const Edge& e ) {
            const float w = shared.weight( e.a, e.b, e.weight );
            if ( w <= max_weight ) {
                candidates.push_back(
                    MREdge{ .weight = w, .lower_bound = e.weight, .a = e.a, .b = e.b } );
            }
        };
        for ( const Edge& e : buffer ) {
            consider( e );
        }
        for ( const Edge& e : evicted ) {
            consider( e );
        }
        buffer.clear();
        evicted.clear();

        // --- 4. merge ---------------------------------------------------------
        sort_edges_by_weight( candidates, sort_scratch );
        merged.clear();
        kruskal_merge( local_tree, candidates, dsu, merged );
        local_tree.swap( merged );
        candidates.clear();

        if ( local_tree.size() != n - 1 ) {
            // The reweighted tree is among the inputs of the merge, so a
            // spanning tree in gives a spanning tree out.
            throw std::runtime_error( "pair_forest_emst_mutual_reachability: a spanning tree in "
                                      "must give a spanning tree out" );
        }

        cutoff = std::min( cutoff, cutoff_from( local_tree ) );

        seconds_spent +=
            std::chrono::duration<double>( std::chrono::steady_clock::now() - start ).count();
    }

    //! Step 3 of the mutual-reachability search: seeds the core distances with
    //! the pairs colliding at the finest prefix of the first few repetitions --
    //! the closest pairs the index knows of. Mirrors
    //! `EMST::seed_core_distances`.
    //!
    //! Unlike there, the pairs are thresholded, at the largest current core
    //! estimate: a pair at least that far apart cannot enter any neighborhood,
    //! so computing its distance would be wasted. On clustered data a finest
    //! prefix can still hold a large fraction of all pairs, and without the
    //! threshold every one of them is computed. When some core is still
    //! infinite the threshold is too, and nothing is pruned.
    //!
    //! Evicted neighbors are dropped: (N) is only needed for the pairs of the
    //! sweep, and a pair seen here that the sweep needs will be seen again.
    //! Returns the number of distances computed.
    template <typename Dataset, typename Hasher, typename Distance>
    static size_t seed_core_distances( const PairForestIndex<Dataset, Hasher, Distance>& index,
                                       SharedCoreDistances& shared ) {
        using ForestIndex = PairForestIndex<Dataset, Hasher, Distance>;
        Timer _t( "seed core distances" );
        if ( !shared.has_cores() ) {
            return 0;
        }
        const size_t reps = std::min<size_t>( 4, index.num_repetitions() );
        const float threshold = shared.max_core();
        size_t distances = 0;
        ParallelExceptionGuard guard;
#pragma omp parallel for num_threads( pair_emst_worker_count( reps ) ) schedule( dynamic, 1 ) \
    reduction( + : distances )
        for ( size_t rep = 0; rep < reps; rep++ ) {
            guard.run( [&] {
                typename ForestIndex::SearchScratch scratch;
                std::vector<Edge> evicted;
                const std::function<bool( std::vector<Edge>& )> sink =
                    [&]( std::vector<Edge>& batch ) {
                        for ( const Edge& e : batch ) {
                            shared.insert( e.a, e.b, e.weight, evicted );
                        }
                        evicted.clear();
                        return false; // scan the whole prefix
                    };
                distances += index.search_pairs( rep,
                                                 ForestIndex::K,
                                                 threshold,
                                                 ForestIndex::COLLECT_BATCH_EDGES,
                                                 sink,
                                                 scratch );
            } );
        }
        guard.rethrow_if_failed();
        // clang-format off
        LOG_INFO( "msg", "seeded core distances",
                  "repetitions", reps,
                  "threshold", threshold,
                  "distances", distances );
        // clang-format on
        return distances;
    }

    //! Step 4 of the mutual-reachability search: one batch, as `run_batch`.
    //!
    //! `best` and `components` are read-only for the batch, as in the plain
    //! search; the core distances are not, they are shared and written by
    //! every thread (see `SharedCoreDistances`). Once the parallel region is
    //! over they are quiescent, and the batch ends with
    //!
    //!  1. every thread's tree reweighted and re-sorted, since its weights are
    //!     as of that thread's last flush;
    //!  2. the trees reduced to one, by pairwise halving;
    //!  3. `absorb_neighborhoods` over the result and the stored neighbor
    //!     pairs whose weight dropped during the batch.
    //!
    //! `best` itself is not merged back in: every thread's tree started as a
    //! copy of it and only ever improved, so the reduction already accounts
    //! for it.
    //!
    //! Returns the new tree, reweighted and sorted as of the end of the batch.
    template <typename Dataset, typename Hasher, typename Distance>
    static std::vector<MREdge>
    run_batch_mr( const PairForestIndex<Dataset, Hasher, Distance>& index,
                  const std::vector<MREdge>& best,
                  SharedCoreDistances& shared,
                  const uint32_t* components,
                  uint8_t k,
                  size_t begin,
                  size_t width,
                  size_t buffer_edges,
                  const PairMrEmstHooks& hooks,
                  size_t& distances_computed ) {
        using ForestIndex = PairForestIndex<Dataset, Hasher, Distance>;

        const size_t n = index.num_points();
        const size_t num_threads = pair_emst_worker_count( width );

        std::vector<std::vector<MREdge>> forests( num_threads );
        size_t batch_distances = 0;
        double enumerate_seconds = 0.0;
        double flush_seconds = 0.0;
        ParallelExceptionGuard guard;

#pragma omp parallel num_threads( num_threads ) reduction( + : batch_distances ) \
    reduction( + : enumerate_seconds ) reduction( + : flush_seconds )
        {
            /// As in `run_batch`: per-thread state, allocated under `guard`.
            std::vector<MREdge> local_tree;
            std::vector<MREdge> merged;
            std::vector<MREdge> candidates;
            std::vector<MREdge> sort_scratch;
            std::vector<Edge> evicted;
            DSU dsu( 0 );
            typename ForestIndex::SearchScratch scratch;
            float cutoff = 0.0f;
            std::function<bool( std::vector<Edge>& )> flush;

            bool healthy = guard.run( [&] {
                /// The sizes `mr_buffer_edges_within_budget` accounted for.
                const size_t tile_edges =
                    static_cast<size_t>( PairCompactTree::TILE_SIZE ) * PairCompactTree::TILE_SIZE;
                const size_t buffered = buffer_edges + tile_edges;
                scratch.tile_buffer.reserve( buffered );
                evicted.reserve( 2 * buffered );
                candidates.reserve( n + 3 * buffered );
                sort_scratch.reserve( n + 3 * buffered );
                merged.reserve( n - 1 );
                local_tree = best;
                dsu = DSU( static_cast<uint32_t>( n ) );
                cutoff = cutoff_from( local_tree );
                flush = [&]( std::vector<Edge>& buffer ) {
                    flush_buffer_mr( buffer,
                                     shared,
                                     local_tree,
                                     merged,
                                     candidates,
                                     sort_scratch,
                                     evicted,
                                     dsu,
                                     cutoff,
                                     n,
                                     hooks,
                                     flush_seconds );
                    return false;
                };
            } );

            const auto thread_start = std::chrono::steady_clock::now();
#pragma omp for schedule( dynamic, 1 )
            for ( size_t rep = begin; rep < begin + width; rep++ ) {
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
        enumerate_seconds -= flush_seconds;

        /// From here on no thread writes the core distances.
        const auto reduce_start = std::chrono::steady_clock::now();
        {
            ParallelExceptionGuard reweight_guard;
#pragma omp parallel for schedule( dynamic, 1 )
            for ( size_t i = 0; i < forests.size(); i++ ) {
                reweight_guard.run( [&] {
                    std::vector<MREdge> sort_scratch;
                    reweight_and_sort( forests[i], shared, sort_scratch );
                } );
            }
            reweight_guard.rethrow_if_failed();
        }
        std::vector<MREdge> reduced = reduce_forests( forests, n );
        std::vector<MREdge> next =
            absorb_neighborhoods( std::move( reduced ), shared, /*only_lowered=*/true, n );
        const double reduce_seconds =
            std::chrono::duration<double>( std::chrono::steady_clock::now() - reduce_start )
                .count();

        // clang-format off
        LOG_INFO( "logger", "pair-emst-mr",
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
        return next;
    }

    //! `tree` as the plain `Edge`s the stopping rule reads, weighted by their
    //! mutual-reachability weight (in raw `Distance` units, like every weight
    //! `stopping_condition` takes).
    static std::vector<Edge> mr_weighted_edges( const std::vector<MREdge>& tree ) {
        std::vector<Edge> out;
        out.reserve( tree.size() );
        for ( const MREdge& e : tree ) {
            out.push_back( Edge{ .weight = e.weight, .a = e.a, .b = e.b } );
        }
        return out;
    }

    //! Approximate minimum spanning tree over `data` under the
    //! mutual-reachability distance with `num_neighbors` neighbors, to within a
    //! factor `1 + epsilon` with probability at least `1 - delta`. The
    //! counterpart, over `PairForestIndex`, of
    //! `EMST::find_tree_mutual_reachability_distance`; see the section comment
    //! above for how it differs.
    //!
    //! The steps are those of `pair_forest_emst`, plus the core distances:
    //!
    //!  1. the seed tree;
    //!  2. the index, and the buffer sizes;
    //!  3. `CoreDistances::random`, sharpened with the seed tree's edges and
    //!     then with the closest pairs of the index (`seed_core_distances`);
    //!  4. `refine_iterations` rounds of NN-descent (`CoreDistances::refine`),
    //!     before any sharing starts;
    //!  5. the initial tree: `absorb_neighborhoods` over the seed tree;
    //!  6. the batches, with the stopping rule applied to the
    //!     mutual-reachability weights. Its confirmed prefix is the tree edges
    //!     no heavier than the confirmed distance `r`: with high probability
    //!     every pair closer than `r` has been seen, so every point whose true
    //!     core is at most `r` has an exact estimate, and every edge of weight
    //!     at most `r` is present with its exact weight. For the same reason
    //!     the confirmed-components filter is safe: both endpoints of a
    //!     confirmed edge have exact cores, so a pair skipped inside a
    //!     component could not have improved one.
    //!
    //! `num_neighbors == 0` gives the plain Euclidean tree, and is handed to
    //! `pair_forest_emst` outright: with no cores to track, the bookkeeping
    //! here would only cost time (16-byte `MREdge`s and a reweighting pass per
    //! flush). The hooks are not called in that case. `num_neighbors >=
    //! n` makes every core distance infinite -- no point has that many
    //! neighbors -- so every spanning tree is minimum, with infinite weight;
    //! the seed tree is returned without searching, since no sweep could ever
    //! confirm an infinite edge.
    //!
    //! Throws `std::invalid_argument` on fewer than two points, and
    //! `std::runtime_error` when the sweep is exhausted without the stopping
    //! rule firing.
    template <typename Dataset, typename Hasher, typename Distance>
    PairMrEmstResult pair_forest_emst_mutual_reachability( const Dataset& data,
                                                           size_t num_neighbors,
                                                           float epsilon,
                                                           float delta,
                                                           size_t repetitions,
                                                           typename Hasher::Builder builder,
                                                           size_t refine_iterations = 10,
                                                           const PairMrEmstHooks& hooks = {} ) {
        Timer _t( "pair-forest-emst-mr" );
        using ForestIndex = PairForestIndex<Dataset, Hasher, Distance>;

        const size_t n = data.size();
        if ( n < 2 ) {
            throw std::invalid_argument(
                "pair_forest_emst_mutual_reachability: needs at least two points" );
        }
        const float delta_per_pair = delta / static_cast<float>( n - 1 );

        if ( num_neighbors == 0 ) {
            PairEmstResult plain = pair_forest_emst<Dataset, Hasher, Distance>(
                data, epsilon, delta, repetitions, std::move( builder ) );
            return PairMrEmstResult{ .tree = std::move( plain.tree ),
                                     .core_distances = CoreDistances( n, 0 ),
                                     .weight = plain.weight,
                                     .distances_computed = plain.distances_computed,
                                     .prefix_at_stop = plain.prefix_at_stop,
                                     .repetitions_at_stop = plain.repetitions_at_stop,
                                     .index_bytes = plain.index_bytes };
        }

        // --- 1. seed ----------------------------------------------------------
        const std::vector<Edge> seed = seed_tree<Dataset, Distance>( data );

        if ( num_neighbors >= n ) {
            CoreDistances cores( n, num_neighbors );
            std::vector<Edge> tree = seed;
            for ( Edge& e : tree ) {
                cores.update( e.a, e.b, e.weight );
                e.weight = std::numeric_limits<float>::infinity();
            }
            std::sort( tree.begin(), tree.end() );
            return PairMrEmstResult{ .tree = std::move( tree ),
                                     .core_distances = std::move( cores ),
                                     .weight = std::numeric_limits<float>::infinity(),
                                     .distances_computed = 0,
                                     .prefix_at_stop = 0,
                                     .repetitions_at_stop = 0,
                                     .index_bytes = 0 };
        }

        // --- 2. index, built once ---------------------------------------------
        const ForestIndex index = build_index<Dataset, Hasher, Distance>(
            data, std::move( builder ), repetitions, seed, delta_per_pair );
        // clang-format off
        LOG_INFO( "msg", "pair forest index constructed",
                  "L", index.num_repetitions(),
                  "K", static_cast<size_t>( ForestIndex::K ),
                  "num_data", n,
                  "num_neighbors", num_neighbors,
                  "delta", delta,
                  "epsilon", epsilon,
                  "family", index.describe_family(),
                  "index_size_Gbytes", static_cast<double>( index.memory_usage() ) / ( 1 << 30 ) );
        // clang-format on

        /// Sized after the index is built, as in `pair_forest_emst`, but
        /// *before* the core distances are allocated: the budget charges them,
        /// and would otherwise count them twice.
        const size_t max_threads = pair_emst_worker_count( PAIR_EMST_BATCH_REPETITIONS );
        const size_t memory_budget = static_cast<size_t>(
            PAIR_EMST_MEMORY_FRACTION * static_cast<double>( available_memory_bytes() ) );
        const size_t buffer_edges =
            mr_buffer_edges_within_budget( n, num_neighbors, max_threads, memory_budget );
        // clang-format off
        LOG_INFO( "msg", "per-thread edge buffers sized",
                  "threads", max_threads,
                  "buffer_edges", buffer_edges,
                  "buffer_edges_per_point", static_cast<double>( buffer_edges ) / n,
                  "memory_budget_Gbytes", static_cast<double>( memory_budget ) / ( 1 << 30 ) );
        // clang-format on

        // --- 3. core distances: random, the seed tree, the index --------------
        CoreDistances initial_cores =
            CoreDistances::random<Dataset, Distance>( data, num_neighbors );
        for ( const Edge& e : seed ) {
            initial_cores.update( e.a, e.b, e.weight );
        }
        SharedCoreDistances shared( std::move( initial_cores ) );
        size_t distances_computed =
            hooks.seed_from_index ? seed_core_distances( index, shared ) : 0;

        // --- 4. NN-descent, before any sharing --------------------------------
        shared.exclusive().template refine<Dataset, Distance>( data, refine_iterations );
        shared.sync_mirror();
        shared.symmetrize();

        // --- 5. the initial tree ----------------------------------------------
        std::vector<MREdge> best;
        {
            best.reserve( n - 1 );
            for ( const Edge& e : seed ) {
                best.push_back(
                    MREdge{ .weight = e.weight, .lower_bound = e.weight, .a = e.a, .b = e.b } );
            }
            if ( hooks.on_start ) {
                hooks.on_start( seed, shared.exclusive() );
            }
            std::vector<MREdge> sort_scratch;
            reweight_and_sort( best, shared, sort_scratch );
            /// The first pass looks at every stored pair: the lowered flags
            /// only describe what changed since a previous pass.
            best = absorb_neighborhoods( std::move( best ), shared, /*only_lowered=*/false, n );
        }

        DSU confirmed( static_cast<uint32_t>( n ) );
        size_t confirmed_edges = 0;
        std::vector<uint32_t> components( n );

        // --- 6. sweep prefixes from the longest to the shortest ---------------
        for ( uint8_t k = ForestIndex::K; k >= 1; k-- ) {
            for ( size_t begin = 0; begin < repetitions; begin += PAIR_EMST_BATCH_REPETITIONS ) {
                const size_t width = std::min( PAIR_EMST_BATCH_REPETITIONS, repetitions - begin );

                confirmed.compress_all();
                for ( size_t i = 0; i < n; i++ ) {
                    components[i] = confirmed.get_parent( static_cast<uint32_t>( i ) );
                }
                const uint32_t* component_filter =
                    ( confirmed_edges > 0 ) ? components.data() : nullptr;

                best = run_batch_mr<Dataset, Hasher, Distance>( index,
                                                                best,
                                                                shared,
                                                                component_filter,
                                                                k,
                                                                begin,
                                                                width,
                                                                buffer_edges,
                                                                hooks,
                                                                distances_computed );
                if ( hooks.on_batch ) {
                    hooks.on_batch( best, shared.exclusive() );
                }

                const size_t repetitions_done = begin + width;
                const std::vector<Edge> weighted = mr_weighted_edges( best );
                const PairEmstStop stop = check_stopping<Dataset, Hasher, Distance>(
                    index, weighted, epsilon, delta_per_pair, k, repetitions_done );

                if ( stop.should_stop ) {
                    // clang-format off
                    LOG_INFO( "msg", "tree found",
                              "prefix", static_cast<size_t>( k ),
                              "repetitions", repetitions_done,
                              "distances_computed", distances_computed );
                    // clang-format on
                    /// The reported weights, as `EMST::find_tree_mutual_
                    /// reachability_distance` computes them. `to_euclidean`
                    /// is monotone, so this is `to_euclidean` of the weight
                    /// the stopping rule summed, and `stop.info.total_weight`
                    /// is their sum.
                    std::vector<Edge> tree;
                    tree.reserve( best.size() );
                    for ( const MREdge& e : best ) {
                        const float w =
                            std::max( { Distance::to_euclidean( e.lower_bound ),
                                        Distance::to_euclidean( shared.core( e.a ) ),
                                        Distance::to_euclidean( shared.core( e.b ) ) } );
                        tree.push_back( Edge{ .weight = w, .a = e.a, .b = e.b } );
                    }
                    std::sort( tree.begin(), tree.end() );
                    return PairMrEmstResult{ .tree = std::move( tree ),
                                             .core_distances = shared.release(),
                                             .weight = stop.info.total_weight,
                                             .distances_computed = distances_computed,
                                             .prefix_at_stop = static_cast<size_t>( k ),
                                             .repetitions_at_stop = repetitions_done,
                                             .index_bytes = index.memory_usage() };
                }

                confirmed.reset();
                for ( size_t idx = 0; idx < stop.info.confirmed_edges; idx++ ) {
                    const MREdge& e = best.at( idx );
                    confirmed.union_sets( e.a, e.b );
                }
                confirmed.compress_all();
                confirmed_edges = stop.info.confirmed_edges;
            }
        }

        throw std::runtime_error( "Minimum spanning tree not found" );
    }

    //! As above, building the hash builder from the dimensionality of the data.
    template <typename Dataset, typename Hasher, typename Distance>
    PairMrEmstResult pair_forest_emst_mutual_reachability( const Dataset& data,
                                                           size_t num_neighbors,
                                                           float epsilon,
                                                           float delta,
                                                           size_t repetitions,
                                                           size_t refine_iterations = 10,
                                                           const PairMrEmstHooks& hooks = {} ) {
        return pair_forest_emst_mutual_reachability<Dataset, Hasher, Distance>(
            data,
            num_neighbors,
            epsilon,
            delta,
            repetitions,
            typename Hasher::Builder( data.get_dimensions() ),
            refine_iterations,
            hooks );
    }

} // namespace panna
