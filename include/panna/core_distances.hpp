#pragma once

//! Core distances and mutual-reachability edges, shared by the two EMST
//! searches that support HDBSCAN's mutual-reachability distance:
//! `EMST::find_tree_mutual_reachability_distance` (`panna/emst.hpp`) and
//! `pair_forest_emst_mutual_reachability` (`panna/pairemst.hpp`).
//!
//! Everything here was moved verbatim out of `emst.hpp`, with the single
//! addition of `CoreDistances::update_one_sided_evicting`, which reports the
//! neighbor an insertion pushed out. `pairemst.hpp` needs it to keep track of
//! the pairs that stop being stored in a neighborhood.

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <optional>
#include <tuple>
#include <utility>
#include <vector>

#include "panna/data.hpp"
#include "panna/expect.hpp"
#include "panna/logging.hpp"
#include "panna/rand.hpp"
#include "panna/timer.hpp"

namespace panna {

    /// A mutual-reachability distance edge, keeping track of the lower
    /// bound on the distance
    struct MREdge {
        float weight;
        float lower_bound;
        uint32_t a;
        uint32_t b;

        bool is_tight() const {
            return weight == lower_bound;
        }

        Edge as_edge() const {
            return { .weight = lower_bound, .a = a, .b = b };
        }

        friend constexpr inline bool operator<( MREdge l, MREdge r ) {
            return std::tie(l.weight, l.lower_bound, l.a, l.b) < std::tie(r.weight, r.lower_bound, r.a, r.b);
        }

        friend constexpr inline bool operator==( MREdge l, MREdge r ) {
            return std::tie(l.weight, l.lower_bound, l.a, l.b) == std::tie(r.weight, r.lower_bound, r.a, r.b);
        }
    };

    /// Maintains information about the nearest neighbors of each point, to
    /// compute core distances.
    /// Can be updated, but access is not synchronized between threads.
    struct CoreDistances {
    private:
        /// how many points we are managing information about
        size_t num_points;
        /// how many neighbors we keep track of
        size_t num_neighbors;
        /// the information about neighbors. For each point we
        /// maintain num_neighbors neighbors
        std::vector<std::pair<float, uint32_t>> neighbors;

        /// Returns true if the neighborhood of `src` was actually improved.
        ///
        /// When `evicted` is non-null and the insertion pushed a *real* neighbor
        /// out of the neighborhood, that neighbor is stored there. The
        /// placeholders a neighborhood starts with (id `uint32_t` max, distance
        /// infinity) are never reported: they stand for no pair at all.
        bool do_update( uint32_t src,
                        uint32_t dst,
                        float dist,
                        std::optional<std::pair<float, uint32_t>>* evicted = nullptr ) {
            // Given the typical small value for num_neighbors,
            // we simply proceed by a linear scan of the points.
            if ( src == dst ) {
                return false;
            }
            if ( num_neighbors == 0 ) {
                return false;
            }
            const size_t offset = src * num_neighbors;
            const float max_distance = neighbors.at( offset ).first;
            if ( dist < max_distance ) {
                const auto begin = neighbors.begin() + offset;
                const auto end = neighbors.begin() + offset + num_neighbors;
                // remove duplicates
                for (auto i=begin; i!=end; i++) {
                    if (i->second == dst) {
                        return false;
                    }
                }
                std::pop_heap( begin, end );
                auto& slot = neighbors.at( offset + num_neighbors - 1 );
                if ( evicted != nullptr && slot.second != std::numeric_limits<uint32_t>::max() ) {
                    *evicted = slot;
                }
                slot = { dist, dst };
                std::push_heap( begin, end );
                return true;
            }
            return false;
        }

        /// Like `do_update`, but holding the spinlock guarding `src`'s
        /// neighborhood. Used by `refine`, where a single pair improves two
        /// neighborhoods that in general belong to two different threads.
        bool do_update_sync( uint32_t src,
                             uint32_t dst,
                             float dist,
                             std::vector<std::atomic_flag>& locks ) {
            while ( locks[src].test_and_set( std::memory_order_acquire ) ) {
                // spin: the critical section is a scan of num_neighbors entries,
                // far shorter than the distance computation that precedes it.
            }
            const bool updated = do_update( src, dst, dist );
            locks[src].clear( std::memory_order_release );
            return updated;
        }

    public:
        using Iterator = std::vector<std::pair<float, uint32_t>>::const_iterator;
                
        explicit CoreDistances(): CoreDistances( 0, 0 ) {
        }
        explicit CoreDistances( size_t num_points, size_t num_neighbors ):
            num_points( num_points ),
            num_neighbors( num_neighbors ),
            neighbors(
                num_points * num_neighbors,
                { std::numeric_limits<float>::infinity(), std::numeric_limits<uint32_t>::max() } ) {
                LOG_INFO("msg", "create CoreDistances", "num_neighbors", num_neighbors);
        }

        template <typename Dataset, typename Distance>
        static CoreDistances random( const Dataset& data, size_t num_neighbors ) {
            Timer _timer("random core distances");
            CoreDistances self( data.size(), num_neighbors );
            // `sample_k( m, k )` needs `k <= m`, and with fewer than
            // `num_neighbors + 1` candidates there is nothing to sample
            // anyway: every other point is used instead, so that no core
            // distance starts at infinity merely for want of candidates.
            std::vector<size_t> pivots;
            if ( data.size() > 0 && num_neighbors + 1 > data.size() - 1 ) {
                pivots.resize( data.size() );
                std::iota( pivots.begin(), pivots.end(), size_t( 0 ) );
            } else if ( data.size() > 0 ) {
                pivots = sample_k( data.size() - 1, num_neighbors + 1 );
            }

            #pragma omp parallel for
            for ( size_t a = 0; a < self.num_points; a++ ) {
                size_t offset = a * num_neighbors;
                size_t neighbor_idx = 0;
                float farthest = 0.0;
                for ( size_t b : pivots ) {
                    if (neighbor_idx >= num_neighbors) {
                        break;
                    }
                    if ( a != b ) {
                        float dist = Distance::compute( data[a], data[b] );
                        expect(neighbor_idx < num_neighbors);
                        self.neighbors.at(offset + neighbor_idx) = { dist, b };
                        if (dist > farthest) {
                            farthest = dist;
                        }
                        neighbor_idx++;
                    }
                }
            }

            for ( size_t a = 0; a < self.num_points; a++ ) {
                const size_t offset = a * num_neighbors;
                auto begin = self.neighbors.begin() + offset;
                auto end = self.neighbors.begin() + offset + num_neighbors;
                std::make_heap(begin, end);
            }
            return self;
        }

        size_t size() const {
            return num_points;
        }

        size_t get_num_neighbors() const {
            return num_neighbors;
        }

        std::vector<uint32_t> get_neighbors(const uint32_t v) const {
            std::vector<uint32_t> nn;
            nn.reserve(num_neighbors);
            size_t offset = v * num_neighbors;
            for ( size_t i = offset; i < offset + num_neighbors; i++ ) {
                nn.push_back(neighbors.at(i).second);
            }
            return nn;
        }

        const std::vector<std::pair<float, uint32_t>>& all() const {
            return neighbors;
        }

        std::pair<Iterator, Iterator> neighbors_view(const uint32_t v) const {
            size_t offset = v * num_neighbors;
            Iterator begin = neighbors.begin() + offset;
            Iterator end = neighbors.begin() + offset + num_neighbors;
            return {begin, end};
        }

        /// update the neighborhood of both a and b, with dist being
        /// their distance
        void update( uint32_t a, uint32_t b, float dist ) {
            do_update( a, b, dist );
            do_update( b, a, dist );
        }

        /// Update only `src`'s neighborhood with the candidate `dst`, at
        /// distance `dist`. Unlike `update`, this touches a single row of
        /// `neighbors`, hence distinct `src` values can be updated
        /// concurrently without any synchronization.
        void update_one_sided( uint32_t src, uint32_t dst, float dist ) {
            do_update( src, dst, dist );
        }

        /// Like `update_one_sided`, but also reports the neighbor the insertion
        /// pushed out of `src`'s neighborhood, if any, as `(distance, id)`.
        ///
        /// Only a successful insertion evicts anything, and what it evicts is
        /// the heap maximum, so after an eviction `core_distance( src )` is no
        /// larger than the evicted distance -- and, core distances only ever
        /// going down, it stays that way. That is the fact
        /// `pair_forest_emst_mutual_reachability` builds on to know when a pair
        /// stops being stored here.
        ///
        /// Returns whether the neighborhood changed. `evicted` is left alone
        /// when no real neighbor was pushed out. Not synchronized.
        bool update_one_sided_evicting( uint32_t src,
                                        uint32_t dst,
                                        float dist,
                                        std::optional<std::pair<float, uint32_t>>& evicted ) {
            return do_update( src, dst, dist, &evicted );
        }

        void update( Edge& edge ) {
            update( edge.a, edge.b, edge.weight );
        }

        /// Improve the neighborhoods with `num_iterations` rounds of NN-descent:
        /// a neighbor of my neighbor is a good candidate to be my neighbor, so
        /// each round performs a local join on the neighborhood of every point,
        /// evaluating the distance between pairs of points that are currently
        /// neighbors of a common point.
        ///
        /// Two standard economies keep each round from costing O(n k^2) forever:
        ///
        ///  - the join considers the *union* of the forward neighbors and of the
        ///    reverse ones (points that picked `p` as a neighbor), with the
        ///    reverse contribution capped at `num_neighbors` entries. Reverse
        ///    edges are what let a pair be discovered from either end, which in
        ///    turn makes it safe to lock only the endpoint being written;
        ///  - a neighbor is *new* if the previous round inserted it. A pair of
        ///    old neighbors was already evaluated in an earlier round, so only
        ///    new-new and new-old pairs are joined. Rounds therefore get rapidly
        ///    cheaper, and the loop stops as soon as one finds no improvement.
        ///
        /// Like `update`, this only ever inserts real points at their real
        /// distances, so a stored k-th neighbor distance can only move *down*
        /// toward its true value: the core distances stay valid upper bounds.
        ///
        /// Returns the total number of accepted neighbor updates.
        template <typename Dataset, typename Distance>
        size_t refine( const Dataset& data, size_t num_iterations ) {
            if ( num_points == 0 || num_neighbors == 0 || num_iterations == 0 ) {
                return 0;
            }
            Timer _timer( "refine core distances" );
            constexpr uint32_t empty = std::numeric_limits<uint32_t>::max();

            std::vector<std::atomic_flag> locks( num_points );
            // The neighborhoods as they were before the previous round's join,
            // against which neighbors are classified as new or old. Empty on the
            // first round, where every neighbor counts as new.
            std::vector<std::pair<float, uint32_t>> previous;

            std::vector<std::vector<uint32_t>> fresh( num_points ), stale( num_points );
            std::vector<uint32_t> reverse_count( num_points );
            size_t total_updates = 0;

            for ( size_t iteration = 0; iteration < num_iterations; iteration++ ) {
                for ( size_t i = 0; i < num_points; i++ ) {
                    fresh[i].clear();
                    stale[i].clear();
                    reverse_count[i] = 0;
                }

                for ( size_t p = 0; p < num_points; p++ ) {
                    const size_t offset = p * num_neighbors;
                    for ( size_t j = 0; j < num_neighbors; j++ ) {
                        const uint32_t nbr = neighbors.at( offset + j ).second;
                        if ( nbr == empty ) {
                            continue;
                        }
                        bool is_new = true;
                        if ( !previous.empty() ) {
                            for ( size_t j2 = 0; j2 < num_neighbors; j2++ ) {
                                if ( previous.at( offset + j2 ).second == nbr ) {
                                    is_new = false;
                                    break;
                                }
                            }
                        }
                        auto& lists = is_new ? fresh : stale;
                        lists.at( p ).push_back( nbr );
                        if ( reverse_count.at( nbr ) < num_neighbors ) {
                            lists.at( nbr ).push_back( static_cast<uint32_t>( p ) );
                            reverse_count.at( nbr )++;
                        }
                    }
                }

                // Snapshot the neighborhoods *before* the join, so that the next
                // iteration classifies exactly this iteration's insertions as new.
                previous = neighbors;

                size_t updates = 0;
                size_t computed = 0;
                #pragma omp parallel for schedule( dynamic, 64 ) reduction( +: updates, computed )
                for ( size_t p = 0; p < num_points; p++ ) {
                    auto& new_list = fresh.at( p );
                    auto& old_list = stale.at( p );
                    // the same point can reach `p` both as a neighbor and as a
                    // reverse neighbor, and joining it with itself is wasted work
                    std::sort( new_list.begin(), new_list.end() );
                    new_list.erase( std::unique( new_list.begin(), new_list.end() ),
                                    new_list.end() );
                    std::sort( old_list.begin(), old_list.end() );
                    old_list.erase( std::unique( old_list.begin(), old_list.end() ),
                                    old_list.end() );

                    auto join = [&]( uint32_t u, uint32_t v ) {
                        if ( u == v ) {
                            return;
                        }
                        const float dist = Distance::compute( data[u], data[v] );
                        computed++;
                        updates += do_update_sync( u, v, dist, locks );
                        updates += do_update_sync( v, u, dist, locks );
                    };

                    for ( size_t i = 0; i < new_list.size(); i++ ) {
                        for ( size_t j = i + 1; j < new_list.size(); j++ ) {
                            join( new_list[i], new_list[j] );
                        }
                        for ( const uint32_t v : old_list ) {
                            join( new_list[i], v );
                        }
                    }
                }

                total_updates += updates;
                // clang-format off
                LOG_INFO( "msg", "nn-descent iteration",
                          "iteration", iteration,
                          "distances", computed,
                          "updates", updates );
                // clang-format on
                if ( updates == 0 ) {
                    break;
                }
            }

            LOG_INFO( "msg", "refined core distances", "updates", total_updates );
            return total_updates;
        }

        void diff(const CoreDistances & other, std::vector<Edge> & out) const {
            for (size_t i=0; i<num_points; i++) {
                if (this->core_distance(i) < other.core_distance(i)) {
                    // there was an improvement in the core distance for point
                    // `i`: collect the neighbor edges that are present in
                    // `this` but not in `other`.
                    auto [this_begin, this_end] = this->neighbors_view(i);
                    auto [other_begin, other_end] = other.neighbors_view(i);
                    for (auto t = this_begin; t != this_end; ++t) {
                        const uint32_t nbr = t->second;
                        // skip empty/sentinel neighbor slots
                        if (nbr == std::numeric_limits<uint32_t>::max()) {
                            continue;
                        }
                        bool in_other = false;
                        for (auto o = other_begin; o != other_end; ++o) {
                            if (o->second == nbr) {
                                in_other = true;
                                break;
                            }
                        }
                        if (!in_other) {
                            out.push_back(
                                Edge{ .weight = t->first,
                                      .a = static_cast<uint32_t>(i),
                                      .b = nbr } );
                        }
                    }
                }
            }
        }

        bool can_improve(const Edge & edge) const {
            return edge.weight <= core_distance( edge.a ) || edge.weight <= core_distance( edge.b );
        }

        /// the distance of the farthest among the num_points
        /// neighbors we keep track of
        float core_distance( uint32_t a ) const {
            const size_t offset = a * num_neighbors;
            return neighbors.at(offset).first;
        }

        /// The current best guess of the mutual reachability
        /// distance between a and b, given the information we accumulated so far.
        /// `dist` is the actual distance between a and b
        float mutual_reachability_distance(uint32_t a, uint32_t b, float dist) const {
            return std::max(std::max(core_distance(a), core_distance(b)),  dist);
        }

        float mutual_reachability_distance( const Edge& e ) const {
            return mutual_reachability_distance( e.a, e.b, e.weight );
        }

        MREdge mutual_reachability_edge( uint32_t a, uint32_t b, float dist ) const {
            float mr_dist = mutual_reachability_distance( a, b, dist );
            return {
                .weight = mr_dist, .lower_bound = dist, .a = a, .b = b
            };
        }

        MREdge mutual_reachability_edge( const Edge& e ) const {
            return mutual_reachability_edge( e.a, e.b, e.weight );
        }
    };

} // namespace panna
