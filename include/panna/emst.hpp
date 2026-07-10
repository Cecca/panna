#pragma once
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <vector>
#include <thread>

#include "panna/billboard.hpp"
#include "panna/channel.hpp"
#include "panna/data.hpp"
#include "panna/dsu.hpp"
#include "panna/logging.hpp"
#include "panna/rand.hpp"
#include "panna/timer.hpp"
#include "panna/trieindex.hpp"
#include "panna/git_version.hpp"

namespace panna {
    // Increment this to signal fundamental changes to
    // the underlying algorithm/implementation
    //
    // Changelog:
    // 13: avoid 0-breakpoints
    // 12: bring the mutual-reachability path on par with find_tree
    //     (seeding, rehash schedule, max-weight pruning)
    // 11: reintroduce rehashing, done differently
    // 10: remove rehashing
    // 9: optimize lattice LSH
    // 8: initialize the solution with a kcenter-based tree
    // 7: refactor parallelism
    // 6: optimize the stopping condition,
    // 5: remove the accumulation of the minimum edge between components,
    //    as it scales quadratically with the number of components
    // 4: hash ancestor data structure, moved the connected components filter
    // 3: use a sample of the dataset for the computation of the scale of the hash
    //    function. All multiplications use AVX instructions
    // 2: unrolled Euclidean distance computation
    // 1: collect additional metrics (memory index and execution profile)
    //    that are available through Python wrapper
    const std::string EMST_VERSION = "13";

    // All weights are euclidean distances, even when the Distance
    // template argument is something else. Conversion is via the
    // to_euclidean method
    struct StoppingConditionInfo {
        const float total_weight;
        const float confirmed_weight;
        const float heaviest_confirmed_edge;
        const size_t edges_to_confirm;
        const size_t confirmed_edges;
    };

    template <typename Edge>
    static void kruskal( DSU& dsu, std::vector<Edge>& edge_list, std::vector<Edge>& output ) {
        for ( const auto& edge : edge_list ) {
            if ( output.size() == dsu.size() - 1 ) {
                break;
            }
            if ( dsu.union_sets( edge.a, edge.b ) ) {
                output.push_back( edge );
            }
        }
    }

    template <typename Dataset, typename Distance>
    std::vector<Edge> random_emst( const Dataset& data ) {
        Timer _t("random_emst");
        const size_t num_data = data.size();
        std::vector<Edge> edges;
        const size_t samples = num_data * std::ceil( std::log10( num_data ) );
        edges.reserve( samples );
        for ( size_t i = 0; i < samples; i++ ) {
            const size_t a = sample_int( 0, num_data - 1 );
            const size_t b = sample_int( 0, num_data - 1 );
            const float d = Distance::compute( data[a], data[b] );
            edges.emplace_back( d, a, b );
        }
        std::sort( edges.begin(), edges.end() );
        std::vector<Edge> res;
        res.reserve( num_data - 1 );
        DSU dsu( num_data );
        kruskal( dsu, edges, res );
        const uint32_t root = 0;
        while ( res.size() < num_data - 1 ) {
            // add arbitrary edges
            for (size_t i=0; i<num_data; i++) {
                if (dsu.union_sets(root, i)) {
                    const float weight = Distance::compute(data[root], data[i]);
                    res.emplace_back(weight, root, i);
                }
            }
        }
        std::sort( res.begin(), res.end() );
        expect( res.size() == num_data - 1 );
        return res;
    }

    template<typename Dataset, typename Distance>
    static std::pair<float, std::vector<Edge>> exact_emst( const Dataset& data ) {
        Timer _t("exact_emst");
        // Compute all the distances
        //  We can pre-allocate all the memory, and avoid the critical region
        const size_t num_data = data.size();
        std::vector<Edge> all_edges( ( num_data - 1 ) * num_data / 2 );
#pragma omp parallel for collapse( 2 )
        for ( size_t i = 0; i < num_data; i++ ) {
            for ( size_t j = i + 1; j < num_data; j++ ) {
                float dist = Distance::compute( data[i], data[j] );
                all_edges.at( i * ( num_data - 1 ) - ( i * ( i + 1 ) / 2 ) + j - 1 ) =
                    Edge{ .weight = dist, .a = (uint32_t)i, .b = (uint32_t)j };
            }
        }
        // Sort the edges
        std::sort( all_edges.begin(), all_edges.end() );
        // Create the DSU
        DSU dsu( num_data );
        float tree_weight = 0;
        std::cout << "Creating the MST" << std::endl;
        std::vector<Edge> tree;
        kruskal( dsu, all_edges, tree );
        expect( tree.size() > 0 );
        LOG_INFO( "msg", "MST created", "heaviest_edge", tree.back().weight );
        for ( const auto& edge : tree ) {
            tree_weight += edge.weight;
        }
        return { tree_weight, tree };
    }

    /// Builds a spanning tree as follows. First the data points are clustered in
    /// std::sqrt(data.size()) clusters with the kcenter algorithm. Then, we compute
    /// the exact EMST of the cluster centers. Finally, we add, for each non-center
    /// point, the edge between itself and its closest cluster center.
    template <typename Dataset, typename Distance>
    std::vector<Edge> clustering_emst( const Dataset& data ) {
        Timer _t("clustering_emst");
        const size_t num_data = data.size();
        const size_t num_clusters = std::ceil( std::sqrt( num_data ) );
        const auto clustering = kcenter<Distance>( data, num_clusters );

        std::vector<Edge> res;
        res.reserve( num_data - 1 );

        // the exact EMST of the centers, with the edge endpoints remapped
        // to the indices of the centers in the original dataset
        const auto [centers_weight, centers_tree] =
            exact_emst<Dataset, Distance>( clustering.centers );
        for ( const auto& edge : centers_tree ) {
            const uint32_t ida = (uint32_t)clustering.center_indices.at( edge.a );
            const uint32_t idb = (uint32_t)clustering.center_indices.at( edge.b );
            if(ida == idb) {
                throw std::runtime_error( "invalid edge!" );
            }
            res.emplace_back( edge.weight, ida, idb );
        }

        // connect each non-center point to its closest center
        std::vector<bool> is_center( num_data, false );
        for ( const size_t c : clustering.center_indices ) {
            is_center.at( c ) = true;
        }
        for ( size_t i = 0; i < num_data; i++ ) {
            if ( !is_center.at( i ) ) {
                res.emplace_back(
                    clustering.distances.at( i ),
                    (uint32_t)i,
                    (uint32_t)clustering.center_indices.at( clustering.assignment.at( i ) ) );
            }
        }

        std::sort( res.begin(), res.end() );
        expect( res.size() == num_data - 1 );
        return res;
    }

    /// `weights` must be sorted in ascending order
    static std::vector<float> find_breaks( const std::vector<float>& weights, float step ) {
        std::vector<float> breaks;
        breaks.push_back( weights.back() );
        LOG_INFO( "weight-break-point", breaks.back() );
        for ( int32_t i = weights.size() - 1; i >= 0; i-- ) {
            const float w = weights[i];
            if (w == 0.0) {
                break;
            }
            if ( w < breaks.back() / step ) {
                LOG_INFO( "weight-break-point", w );
                breaks.push_back( w );
            }
        }
        std::reverse(breaks.begin(), breaks.end());
        return breaks;
    }

    static std::vector<float> find_breaks( const std::vector<Edge>& tree, float step ) {
        std::vector<float> weights;
        weights.reserve( tree.size() );
        for ( const auto& e : tree ) {
            weights.push_back( e.weight );
        }
        return find_breaks( weights, step );
    }

    /// A unit of work pulled by a persistent worker: one repetition at one prefix.
    /// The worker pool is spawned once per find_tree call and loops on a single
    /// channel of these across all prefixes.
    struct WorkItem {
        size_t prefix;
        size_t repetition;
    };

    struct MRPartial {
        std::vector<Edge> tree_edges;
        std::vector<Edge> core_distance_edges;
    };

    /// The tentative minimm spanning tree as it is being constructed
    /// by multiple threads
    struct RunningResult {
        std::vector<Edge> tree;
        DSU filter;

        explicit RunningResult(): tree(), filter( 0 ) {
        }
        explicit RunningResult( std::vector<Edge>&& tree, DSU&& filter ):
            tree( std::move( tree ) ), filter( std::move( filter ) ) {
        }
    };

    /// Working state owned by the collector loop of `find_tree`. Constructed once
    /// and mutated in place across partials so the per-partial copies and
    /// allocations (full tree copy, fresh DSUs, fresh merge buffer) are avoided.
    /// The collector is the sole writer of the `Billboard`, so this state stays in
    /// sync with the published snapshot.
    struct ReducerState {
        std::vector<Edge> tree;        // clean found forest (real edges only)
        std::vector<Edge> completed;   // scratch: found forest + arbitrary completion
        DSU union_find;        // per-partial merge DSU
        DSU completion_filter; // mirror of union_find for complete_arbitrarily
        DSU confirmed_filter;  // published filter; refreshed only on decide partials
        std::vector<Edge> merge_scratch;

        explicit ReducerState( uint32_t n ):
            tree(), completed(), union_find( n ), completion_filter( n ),
            confirmed_filter( n ), merge_scratch() {
        }
    };

    /// Simulate a run of Kruskal's algorithm, assuming both input vectors are sorted.
    /// Report in the output vector the edges from `new_edges` that would be
    /// part of the updated tree.
    static void kruskal_new_edges( const std::vector<Edge>& old_edges,
                                   const std::vector<Edge>& new_edges,
                                   DSU& union_find,
                                   std::vector<Edge>& out ) {
        expect( std::is_sorted( old_edges.begin(), old_edges.end() ) );
        expect( std::is_sorted( new_edges.begin(), new_edges.end() ) );

        union_find.reset();
        size_t asize = old_edges.size();
        size_t bsize = new_edges.size();
        size_t aidx = 0;
        size_t bidx = 0;

        while ( aidx < asize && bidx < bsize ) {
            if ( old_edges.at( aidx ) < new_edges.at( bidx ) ) {
                auto e = old_edges.at(aidx++);
                union_find.union_sets( e.a, e.b );
            } else {
                auto e = new_edges.at(bidx++);
                if ( union_find.union_sets( e.a, e.b ) ) {
                    out.push_back( e );
                }
            }
        }
        while ( aidx < asize ) {
            auto e = old_edges.at(aidx++);
            union_find.union_sets( e.a, e.b );
        }
        while ( bidx < bsize ) {
            auto e = new_edges.at(bidx++);
            if ( union_find.union_sets( e.a, e.b ) ) {
                out.push_back( e );
            }
        }
    }


    /// implementation of Kruskal's algorithm that picks updates from two sorted
    /// vectors. Avoids having to sort both their concatenation.
    static void kruskal_merge( const std::vector<Edge>& old_edges,
                                   const std::vector<Edge>& new_edges,
                                   DSU& union_find,
                                   std::vector<Edge>& out ) {
        expect( std::is_sorted( old_edges.begin(), old_edges.end() ) );
        expect( std::is_sorted( new_edges.begin(), new_edges.end() ) );

        union_find.reset();
        size_t asize = old_edges.size();
        size_t bsize = new_edges.size();
        size_t aidx = 0;
        size_t bidx = 0;

        while ( aidx < asize && bidx < bsize ) {
            Edge e;
            if ( old_edges.at( aidx ) < new_edges.at( bidx ) ) {
                e = old_edges.at(aidx++);
            } else {
                e = new_edges.at(bidx++);
            }
            if ( union_find.union_sets( e.a, e.b ) ) {
                out.push_back( e );
            }
        }
        while ( aidx < asize ) {
            auto e = old_edges.at(aidx++);
            if ( union_find.union_sets( e.a, e.b ) ) {
                out.push_back( e );
            }
        }
        while ( bidx < bsize ) {
            auto e = new_edges.at(bidx++);
            if ( union_find.union_sets( e.a, e.b ) ) {
                out.push_back( e );
            }
        }
    }

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

        void do_update( uint32_t src, uint32_t dst, float dist ) {
            // Given the typical small value for num_neighbors,
            // we simply proceed by a linear scan of the points.
            if ( src == dst ) {
                return;
            }
            if ( num_neighbors == 0 ) {
                return;
            }
            const size_t offset = src * num_neighbors;
            const float max_distance = neighbors.at( offset ).first;
            if ( dist < max_distance ) {
                const auto begin = neighbors.begin() + offset;
                const auto end = neighbors.begin() + offset + num_neighbors;
                // remove duplicates
                for (auto i=begin; i!=end; i++) {
                    if (i->second == dst) {
                        return;
                    }
                }
                std::pop_heap( begin, end );
                neighbors.at( offset + num_neighbors - 1 ) = { dist, dst };
                std::push_heap( begin, end );
            }
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
            std::vector<size_t> pivots = sample_k( data.size() - 1, num_neighbors + 1 );

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

        void update( Edge& edge ) {
            update( edge.a, edge.b, edge.weight );
        }

        void diff(const CoreDistances & other, std::vector<Edge> & out) const {
            for (size_t i=0; i<num_points; i++) {
                if (this->core_distance(i) < other.core_distance(i)) {
                    // there was an improvement in the core distance for point
                    // `i`: collect the neighbor edges that are present in
                    // `other` but no longer in `this`.
                    auto [this_begin, this_end] = this->neighbors_view(i);
                    auto [other_begin, other_end] = other.neighbors_view(i);
                    for (auto o = other_begin; o != other_end; ++o) {
                        const uint32_t nbr = o->second;
                        // skip empty/sentinel neighbor slots
                        if (nbr == std::numeric_limits<uint32_t>::max()) {
                            continue;
                        }
                        bool in_this = false;
                        for (auto t = this_begin; t != this_end; ++t) {
                            if (t->second == nbr) {
                                in_this = true;
                                break;
                            }
                        }
                        if (!in_this) {
                            out.push_back(
                                Edge{ .weight = o->first,
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

    /// For each step of the algorithm, record some stats
    struct ExecutionProfileElement {
        long elapsed_ms = 0;
        size_t prefix = 0;
        size_t repetition = 0;
        float emst_confirmed_weight = 0.0;
        float emst_weight_lower_bound = 0.0;
        float emst_max_weight = 0.0;
        float emst_max_confirmed_weight = 0.0;
        float emst_total_weight = 0.0;
        size_t emst_num_confirmed = 0;
    };

    struct MRRunningResult {
        std::vector<Edge> tree;
        DSU filter;
        CoreDistances neighborhoods;

        explicit MRRunningResult(): tree(), filter( 0 ), neighborhoods() {
        }
        explicit MRRunningResult( std::vector<Edge>&& tree,
                                  DSU&& filter,
                                  CoreDistances&& neighborhoods ):
            tree( std::move( tree ) ),
            filter( std::move( filter ) ),
            neighborhoods( std::move( neighborhoods ) ) {
        }
    };

    /// Working state owned by the collector loop of
    /// `find_tree_mutual_reachability_distance`. Like `ReducerState`, but carries
    /// an owned `CoreDistances` instead of a completion filter / merge scratch (the
    /// MR merge uses `update_tree`, which manages its own scratch).
    struct MRReducerState {
        std::vector<Edge> tree;
        DSU filter;
        CoreDistances core_distances;

        explicit MRReducerState( uint32_t n ): tree(), filter( n ), core_distances() {
        }
    };

    template <typename Dataset, typename Hasher, typename Distance>
    class EMST {
        uint32_t dimensionality;
        size_t max_repetitions;
        uint32_t max_hashbits;
        public:
        Index<Dataset, Hasher, Distance> table;
        private:
        uint32_t num_data{ 0 };
        float delta{ 0.01 };
        const float epsilon{ 0.2 };
        size_t distances_computed = 0;
        size_t num_collisions = 0;
        size_t index_size_bytes = 0;
        std::vector<ExecutionProfileElement> profile;
        /// spanning tree built with `clustering_emst` at construction time,
        /// used to initialize the search in `find_tree`
        std::vector<Edge> initial_tree;

        static size_t get_worker_count( size_t max_repetitions ) {
            const size_t hw = std::max<size_t>( 1, std::thread::hardware_concurrency() );
            size_t workers = ( hw > 1 ) ? ( hw - 1 ) : 1;

            // Cap default fan-out to avoid memory-bandwidth saturation on high-core hosts.
            workers = std::min( workers, static_cast<size_t>( 32 ) );
            workers = std::min( workers, std::max<size_t>( 1, max_repetitions ) );

            if ( const char* env = std::getenv( "PANNA_EMST_THREADS" ); env != nullptr ) {
                char* end = nullptr;
                const unsigned long parsed = std::strtoul( env, &end, 10 );
                if ( end != env && *end == '\0' && parsed > 0 ) {
                    workers = static_cast<size_t>( parsed );
                    workers = std::min( workers, std::max<size_t>( 1, max_repetitions ) );
                }
            }

            return std::max<size_t>( 1, workers );
        }

    public:
        EMST() {}
        /**
         * @brief Class to construct an approximate Euclidean Mininmum Spanning Tree from data
         * points
         *
         * @param dimensions Dimension of the hash index
         * @param repetitions Number of repetitions for the LSH index
         * @param builder Builder for the hash function
         * @param data_in Input data points
         * @param data_dimensionality Dimensionality of the input data
         * @param delta Probability of failure parameter (default: 0.01)
         * @param epsilon Approximation factor parameter (default: 0.2)
         *
         * @details This constructor initializes an EMST object by:
         * 1. Set up the LSH index table with the distance metric
         * 2. Insert all input vectors into the index
         * 3. Rebuilding the index structure
         * 4. Construct a Union Find data structure
         * The constructor takes ownership of the input data through a move operation.
         */
        EMST( const size_t dimensions,
              const size_t repetitions,
              std::vector<std::vector<float>>& data_in,
              const float delta_in = 0.01f,
              const float epsilon = 0.2f ):
            dimensionality( dimensions ),
            max_repetitions( 0 ),
            max_hashbits( 0 ),
            table( EMST::setup_index( data_in, dimensions, repetitions ) ),
            num_data( data_in.size() ),
            epsilon( epsilon ),
            distances_computed( 0 ),
            num_collisions( 0 ),
            index_size_bytes( 0 ),
            profile() {
            LOG_INFO("git-version", GIT_COMMIT_HASH);

            delta = delta_in;

            // initialize the table
            const Dataset & dataset = table.get_dataset();
            initial_tree = clustering_emst<Dataset, Distance>(dataset);

            // Get info on the index
            max_hashbits = table.num_concatenations();
            max_repetitions = table.num_repetitions();

            // Measure the size of the index
            index_size_bytes = table.memory_usage();
            LOG_INFO("msg", "Index constructed",
                     "L", max_repetitions,
                     "K", max_hashbits,
                     "num_data", num_data,
                     "delta", delta,
                     "index_size_Gbytes", ((double)index_size_bytes )/ (1 << 30));
        };

        /// @brief Destructor
        ~EMST() = default;

        static Index<Dataset, Hasher, Distance>
        setup_index( const std::vector<std::vector<float>>& data_in,
                     size_t dimensions,
                     size_t repetitions ) {
            typename Hasher::Builder builder(dimensions);

            Index<Dataset, Hasher, Distance> table( dimensions, builder, repetitions );
            for ( auto& point : data_in ) {
                table.insert( point.begin(), point.end() );
            }

            return table;
        }

        /// @brief the number of distances actually computed by the algorithm
        size_t get_distance_count() const {
            return distances_computed;
        }

        /// @brief the number of collisions seen by the algorithm
        size_t get_collisions_count() const {
            return num_collisions;
        }

        size_t get_index_size_bytes() const {
            return index_size_bytes;
        }

        std::vector<ExecutionProfileElement> get_profile() const {
            return profile;
        }

        /// Complete the given forest with arbitrary edges so that it becomes a connected tree
        size_t complete_arbitrarily(std::vector<Edge> & forest, DSU& dsu) const {
            // now connect the tree containing `0` with all the other trees
            size_t added_cnt = 0;
            const uint32_t root = 0;
            for (uint32_t i=1; i < num_data && forest.size() < num_data - 1; i++) {
                if (dsu.union_sets(root, i)) {
                    const float weight = table.get_distance(root, i);
                    forest.emplace_back(weight, root, i);
                    added_cnt++;
                }
            }
            std::sort(forest.begin(), forest.end());
            return added_cnt;
        }

        size_t complete_arbitrarily(std::vector<Edge> & forest) const {
            DSU dsu(num_data);
            for (const auto & e : forest) {
                dsu.union_sets(e.a, e.b);
            }
            return complete_arbitrarily(forest, dsu);
        }

        /// @brief Computes the exact MST with Kruskal's algorithm in a naive way
        /// @return weight of the exact MST
        std::pair<float, std::vector<Edge>> exact_tree() {
            return exact_emst<Dataset, Distance>(table.get_dataset());
        }

        std::pair<float, std::vector<Edge>> exact_mutual_reachability_distance_tree( const size_t num_neighbors ) {
            // Clear from any previous runs
            clear();
            // Compute all the distances
            //  We can pre-allocate all the memory, and avoid the critical region
            std::vector<Edge> all_edges( ( num_data - 1 ) * num_data / 2 );
#pragma omp parallel for collapse (2)
            for ( size_t i = 0; i < num_data; i++ ) {
                for ( size_t j = i + 1; j < num_data; j++ ) {
                    float dist = table.get_distance( i, j );
                    all_edges.at(i * ( num_data - 1 ) - ( i * ( i + 1 ) / 2 ) + j - 1) =
                        Edge{ .weight = dist, .a = (uint32_t)i, .b = (uint32_t)j };
                }
            }
            CoreDistances cd( num_data, num_neighbors );
            for (auto &e: all_edges) {
                cd.update(e.a, e.b, e.weight);
            }

            // Create the DSU
            float tree_weight = 0;
            std::cout << "Creating the MST" << std::endl;
            std::vector<Edge> tree;
            update_tree( tree, all_edges, cd );
            for ( const auto& edge : tree ) {
                tree_weight += edge.weight ;
            }
            LOG_INFO("msg", "MST created",
                      "heaviest_edge",  tree.back().weight ,
                      "tree-weight", tree_weight
            );
            // Reweight the output edges with the mutual reachability
            // distance (which was already used in update_tree).
            // Furthermore, switch to the Euclidean distance, if
            // the metric used was something different
            for ( size_t i = 0; i < tree.size(); i++ ) {
                const float w = Distance::to_euclidean( tree[i].weight );
                const float ca =
                    Distance::to_euclidean( cd.core_distance( tree[i].a ) );
                const float cb =
                    Distance::to_euclidean( cd.core_distance( tree[i].b ) );
                tree[i].weight = std::max( { w, ca, cb } );
            }

            return {tree_weight, tree};
        }

        /// the worker function in find_tree. Persistent across prefixes:
        /// it loops pulling WorkItems until the work channel is closed.
        static void worker_fun( const size_t tid,
                                const Index<Dataset, Hasher, Distance> &table,
                                Billboard<RunningResult> &running_result,
                                std::atomic_bool &found,
                                std::atomic<float> &max_weight,
                                std::atomic_size_t &count_distances,
                                std::atomic_size_t &count_collisions,
                                Channel<WorkItem> &work,
                                Channel<std::vector<Edge>> &partials ) {
            for ( std::optional<WorkItem> oitem = work.receive(); oitem.has_value();
                  oitem = work.receive() ) {
                const size_t prefix = oitem->prefix;
                const size_t repetition = oitem->repetition;
                LOG_INFO( "tid", tid, "repetition", repetition, "prefix", prefix, "logger", "worker" );
                Timer _timer("worker-repetition");
                if ( found ) {
                    // Tree already found: skip the work but still send a (empty) partial
                    // so the driver's per-prefix drain count stays balanced.
                    LOG_INFO( "tid", tid, "logger", "worker", "msg", "tree found, skipping work item" );
                    partials.send( std::vector<Edge>() );
                    continue;
                }
                float sum_distances = 0.0, min_distance = std::numeric_limits<float>::infinity(), max_distance = 0.0;
                float avg_denom = 0.0;
                auto rr = running_result.read();
                // Read the published snapshot through the shared_ptr (the collector is
                // the sole mutator). local_tree is only consumed as old_edges by
                // kruskal_new_edges, and filter is only probed via get_parent, so neither
                // needs a copy; the Kruskal scratch DSU is the only worker-local state.
                const std::vector<Edge>& local_tree = rr->tree;
                DSU dsu( rr->filter.size() );
                std::vector<Edge> output;
                std::vector<Edge> candidates;
                candidates.reserve(10*dsu.size());
                auto [cnt_dist, cnt_collisions] = table.search_pairs_different_groups(
                    repetition,
                    prefix,
                    10 * dsu.size(), // buffer size
                    max_weight,
                    [&]( uint32_t x ) { return rr->filter.get_parent( x ); },
                    [&]( std::vector<Edge>& scratch ) {
                        LOG_DEBUG( "msg", "building tree on batch", "logger", "worker", "batch_size", scratch.size() );
                        for ( auto& e : scratch ) {
                            sum_distances += e.weight;
                            if (e.weight < min_distance) {
                                min_distance = e.weight;
                            }
                            if (e.weight > max_distance) {
                                max_distance = e.weight;
                            }
                        }
                        avg_denom += scratch.size();

                        std::sort( scratch.begin(), scratch.end() );
                        kruskal_new_edges(local_tree, scratch, dsu, output);

                        return found.load(); // early stop if the solution has been found in the meantime
                    } );
                float avg_distance = sum_distances / avg_denom;
                // clang-format off
                LOG_INFO("logger", "worker", "tid", tid, "repetition", repetition, "prefix", prefix,
                          "cnt_distances", cnt_dist, "cnt_collisions", cnt_collisions,
                          "average_distance", avg_distance,
                          "min_distance", min_distance,
                          "max_distance", max_distance);
                // clang-format on
                count_distances += cnt_dist;
                count_collisions += cnt_collisions;
                expect(cnt_dist == cnt_collisions);
                std::sort(output.begin(), output.end());
                partials.send( std::move(output) );
            }
        }


        /// Persistent across prefixes: loops pulling WorkItems until
        /// the work channel is closed.
        static void worker_fun_mutual_reachability( const size_t tid,
                                                    const Index<Dataset, Hasher, Distance>& table,
                                                    Billboard<MRRunningResult>& running_result,
                                                    std::atomic_bool& found,
                                                    std::atomic<float>& max_weight,
                                                    std::atomic_size_t& count_distances,
                                                    std::atomic_size_t& count_collisions,
                                                    Channel<WorkItem>& work,
                                                    Channel<MRPartial>& partials ) {
            for ( std::optional<WorkItem> oitem = work.receive(); oitem.has_value();
                  oitem = work.receive() ) {
                const size_t prefix = oitem->prefix;
                const size_t repetition = oitem->repetition;
                Timer _timer("worker-repetition");
                if ( found ) {
                    // Tree already found: skip the work but still send a (empty) partial
                    // so the driver's per-prefix drain count stays balanced.
                    LOG_INFO(
                        "tid", tid, "logger", "worker", "msg", "tree found, skipping work item" );
                    MRPartial partial;
                    partials.send( std::move( partial ) );
                    continue;
                }
                float sum_distances = 0.0, min_distance = std::numeric_limits<float>::infinity(), max_distance = 0.0;
                float avg_denom = 0.0;
                auto rr = running_result.read();
                // local_tree is rebuilt in place by update_tree, so it stays a
                // worker-local copy.
                // The core distances are a worker-local copy as well, since to reduce the
                // memory usage we have to update them.
                std::vector<Edge> local_tree( rr->tree );
                CoreDistances neighborhoods(rr->neighborhoods);
                LOG_INFO(
                    "tid", tid, "repetition", repetition, "prefix", prefix, "logger", "worker" );
                // The edges we have to keep even if they are not part of the tree,
                // because they might be updated to a smaller weight in the future
                std::vector<MREdge> non_tree_edges;
                auto [cnt_dist, cnt_collisions] = table.search_pairs_different_groups(
                    repetition,
                    prefix,
                    10 * rr->filter.size(), // buffer size
                    max_weight, // TODO: watch out this line
                    [&]( uint32_t x ) { return rr->filter.get_parent( x ); },
                    [&]( std::vector<Edge>& updates ) {
                        // add to the possibly useful edges only if they would
                        // improve the local copy of the core distances. The alternative
                        // is to just accumulate all possibly improving edges.
                        for (auto & e : updates) {
                            sum_distances += e.weight;
                            if (e.weight < min_distance) {
                                min_distance = e.weight;
                            }
                            if (e.weight > max_distance) {
                                max_distance = e.weight;
                            }
                            // TODO: here the edges that are _not_ inserted
                            // in the core distances are the ones that can
                            // actually participate in the tree?
                            neighborhoods.update(e);
                        }
                        avg_denom += updates.size();
                        LOG_INFO("logger", "worker", "tid", tid, "repetition", repetition,
                                 "prefix", "prefix", "updates-size", updates.size());
                        update_tree( local_tree, updates, neighborhoods );
                        // updates.clear();
                        expect( local_tree.size() > 0 );
                        // early stop if the solution has been found in the meantime
                        return found.load();
                    } );
                float avg_distance = sum_distances / avg_denom;
                // clang-format off
                LOG_INFO("logger", "worker", "tid", tid, "repetition", repetition, "prefix", prefix,
                          "cnt_distances", cnt_dist, "cnt_collisions", cnt_collisions,
                          "average_distance", avg_distance,
                          "min_distance", min_distance,
                          "max_distance", max_distance);
                // clang-format on
                count_distances += cnt_dist;
                count_collisions += cnt_collisions;
                MRPartial partial;
                // std::vector<Edge> possibly_useful_edges;
                neighborhoods.diff(rr->neighborhoods, partial.core_distance_edges);
                // clang-format off
                LOG_INFO("logger", "worker", "tid", tid, "repetition", repetition,
                         "prefix", "prefix", "core-distances-diff", partial.core_distance_edges.size());
                // clang-format on
                partial.tree_edges = std::move( local_tree );
                // TODO: send core distance edges and tree edges separately
                partials.send( std::move( partial ) );
            }
        }

        /// find the minimum spanning tree, using channels to handle parallelism
        std::pair<float, std::vector<Edge>> find_tree() {
            clear();
            const auto find_start_t = std::chrono::steady_clock::now();

            std::vector<float> breaks;
            if constexpr ( Hasher::Builder::fits_to_distance ) {
                breaks = find_breaks( initial_tree, 10.0 );
            } else {
                // the builder ignores the fitting distance,
                // so a single arbitrary break suffices
                breaks = { initial_tree.back().weight };
            }

            // Start the search from the clustering-based spanning tree built at
            // construction time rather than from an empty forest. The confirmed
            // filter stays empty (no edge is confirmed yet), but the heaviest
            // edge of the initial tree already bounds the weight of any MST edge
            // (cycle property), so workers can prune distances right away.
            Billboard<RunningResult> running_result;
            running_result.update(
                RunningResult( std::vector<Edge>( initial_tree ), DSU( num_data ) ) );

            // Collector-owned working state, constructed once and kept in sync with
            // the published snapshot across prefixes.
            ReducerState state( num_data );
            state.tree = initial_tree;

            // The expensive completion + stopping check runs on a cadence rather
            // than on every partial. Larger values cut serial collector work but
            // may run a few extra repetitions before a valid stop is noticed.
            // Gating only ever *delays* stopping, so the (1+epsilon) guarantee holds.
            constexpr size_t DECIDE_PERIOD = 16;

            std::atomic<float> max_weight( initial_tree.empty()
                                               ? std::numeric_limits<float>::infinity()
                                               : initial_tree.back().weight );
            float tree_weight = 0;
            std::atomic_size_t count_distances( 0 ), count_collisions( 0 );
            const size_t max_threads = get_worker_count( max_repetitions );
            LOG_INFO( "msg",
                      "parallelism config",
                      "worker_threads",
                      max_threads,
                      "max_repetitions",
                      max_repetitions );

            std::atomic_bool found( false );

            // Persistent worker pool: spawned once and reused across every prefix
            // so threads are created max_threads times total (not
            // max_threads x prefixes). Workers loop on these channels
            // until `work` is closed at shutdown.
            Channel<WorkItem> work( max_repetitions );
            Channel<std::vector<Edge>> partials( max_repetitions );
            std::vector<std::thread> workers;
            for ( size_t tid = 0; tid < max_threads; tid++ ) {
                workers.emplace_back( EMST::worker_fun,
                                      tid,
                                      std::ref( table ),
                                      std::ref( running_result ),
                                      std::ref( found ),
                                      std::ref( max_weight ),
                                      std::ref( count_distances ),
                                      std::ref( count_collisions ),
                                      std::ref( work ),
                                      std::ref( partials ) );
            }

            bool first_build = true;
            for ( const float distance_break : breaks ) {
                if (distance_break == 0.0) {
                    throw std::runtime_error("invalid distance break 0.0");
                }
                if (found.load()) {
                    break;
                }
                LOG_INFO( "distance-break", distance_break );
                table.builder.reset();
                table.builder.fit( table.get_dataset(),
                                   distance_break,
                                   table.num_repetitions(),
                                   delta / ( num_data - 1 ) );

                size_t initial_prefix = max_hashbits;
                if (first_build) {
                    // In this case the last hash value holds the smallest hash value of
                    // the previous repetition
                    table.rebuild();
                    first_build = false;
                } else {
                    table.rehash();
                    initial_prefix--;
                }
                for ( size_t prefix = initial_prefix; prefix > 0 && !found; prefix-- ) {
                    // Enqueue this prefix's repetitions for the persistent pool.
                    for ( size_t repetition = 0; repetition < max_repetitions; repetition++ ) {
                        work.send( WorkItem{ .prefix = prefix, .repetition = repetition } );
                    }

                    // Drain exactly one partial per enqueued item. We always drain the
                    // whole prefix (even after `found`) so that no work item is left in
                    // the channel and every worker returns to waiting on `work.receive`
                    // -- this is what guarantees index quiescence.
                    size_t completed_repetitions = 0;
                    while ( completed_repetitions < max_repetitions ) {
                        std::optional<std::vector<Edge>> local_tree = partials.receive();
                        expect( local_tree.has_value() );
                        completed_repetitions++;
                        if ( found ) {
                            // discard late partials; keep draining to balance the channel
                            continue;
                        }
                        std::vector<Edge> update = std::move( *local_tree );
                        // clang-format off
                        LOG_INFO( "logger", "collector",
                                  "msg", "received update",
                                  "update-size", update.size() );
                        // clang-format on

                        // Merge the incoming partial into the owned tree using the
                        // reused scratch buffer, then swap. kruskal_merge resets the
                        // persistent union_find internally.
                        state.merge_scratch.clear();
                        kruskal_merge( state.tree, update, state.union_find, state.merge_scratch );
                        std::swap( state.tree, state.merge_scratch );
                        update.clear();
                        // clang-format off
                        LOG_INFO( "logger", "collector",
                                  "tree-size", state.tree.size(),
                                  "prefix", prefix,
                                  "completed-repetitions", completed_repetitions );
                        // clang-format on

                        // state.tree is the clean found forest; a forest with k edges
                        // over n nodes has n-k components, so this is O(1).
                        const size_t num_components = num_data - state.tree.size();
                        // Run the expensive completion + stopping check on a cadence:
                        // whenever the forest is already spanning (cheap, no
                        // completion), periodically as a fallback, and always on the
                        // last partial of a prefix so we never waste it.
                        const bool do_decide = ( num_components == 1 ) ||
                                               ( completed_repetitions % DECIDE_PERIOD == 0 ) ||
                                               ( completed_repetitions >= max_repetitions );

                        bool publish_completed = false;
                        if ( do_decide ) {
                            // Evaluate the stopping condition on a spanning tree. When the
                            // found forest is already spanning use it directly (no sort);
                            // otherwise complete it into a scratch buffer, leaving
                            // state.tree clean for the next merge / component count.
                            const std::vector<Edge>* eval = &state.tree;
                            if ( num_components > 1 ) {
                                const auto start = std::chrono::steady_clock::now();
                                state.completed = state.tree;
                                state.completion_filter = state.union_find;
                                const size_t added_edges = complete_arbitrarily(
                                    state.completed, state.completion_filter );
                                const auto end = std::chrono::steady_clock::now();
                                const double elapsed_ms =
                                    std::chrono::duration_cast<std::chrono::milliseconds>( end -
                                                                                           start )
                                        .count();
                                if ( added_edges > 0 ) {
                                    // clang-format off
                                    LOG_INFO( "msg", "completed tree with arbitrary edges",
                                              "elapsed_ms", elapsed_ms,
                                              "added_edges", added_edges );
                                    // clang-format on
                                }
                                eval = &state.completed;
                            }

                            // FIXME: redundant, since we completed the tree arbitrarily
                            if ( eval->size() == num_data - 1 ) {
                                StoppingConditionInfo stop =
                                    stopping_condition( *eval, prefix, completed_repetitions );
                                float weight_lower_bound =
                                    stop.confirmed_weight +
                                    stop.edges_to_confirm * stop.heaviest_confirmed_edge;
                                LOG_INFO( "weight-lower-bound", weight_lower_bound );
                                bool should_stop =
                                    stop.total_weight <= ( 1 + epsilon ) * weight_lower_bound;
                                // clang-format off
                                LOG_INFO( "logger", "collector",
                                          "stop.total_weight", stop.total_weight,
                                          "stop.confirmed_weight", stop.confirmed_weight,
                                          "stop.heaviest_confirmed_edge", stop.heaviest_confirmed_edge,
                                          "stop.edges_to_confirm", stop.edges_to_confirm,
                                          "heaviest_edge", eval->at(num_data-2).weight,
                                          "weight_lower_bound", weight_lower_bound,
                                          "should_stop", should_stop );
                                // clang-format on
                                max_weight = eval->back().weight;
                                float mean_weight = 0.0;
                                for ( auto& e : *eval ) {
                                    mean_weight += e.weight;
                                }
                                mean_weight /= eval->size();
                                LOG_INFO( "logger",
                                          "collector",
                                          "max-weight",
                                          max_weight.load(),
                                          "mean-weight",
                                          mean_weight );
                                profile.push_back( ExecutionProfileElement{
                                    .elapsed_ms =
                                        std::chrono::duration_cast<std::chrono::milliseconds>(
                                            std::chrono::steady_clock::now() - find_start_t )
                                            .count(),
                                    .prefix = prefix,
                                    .repetition = completed_repetitions,
                                    .emst_confirmed_weight = stop.confirmed_weight,
                                    .emst_weight_lower_bound = weight_lower_bound,
                                    .emst_max_weight = max_weight,
                                    .emst_max_confirmed_weight = stop.heaviest_confirmed_edge,
                                    .emst_total_weight = stop.total_weight,
                                    .emst_num_confirmed = num_data - 1 - stop.edges_to_confirm } );

                                // stop if we are done
                                if ( should_stop ) {
                                    LOG_INFO( "msg", "tree found, signalling stop" );
                                    found = true;
                                    tree_weight = stop.total_weight;
                                    // if completion was needed, the spanning result lives
                                    // in state.completed; publish that as the final tree.
                                    publish_completed = ( num_components > 1 );
                                }
                                // Refresh the confirmed-edge filter from the evaluated tree.
                                state.confirmed_filter.reset();
                                for ( size_t idx = 0; idx < stop.confirmed_edges; idx++ ) {
                                    auto edge = eval->at( idx );
                                    state.confirmed_filter.union_sets( edge.a, edge.b );
                                }
                                state.confirmed_filter.compress_all();
                            }
                        }
                        // publish the new running result: copy the owned state into the
                        // billboard (the one unavoidable copy, read by the workers). On
                        // non-decide partials we publish the clean found forest and the
                        // carried-forward confirmed filter (a valid, monotone subset).
                        const std::vector<Edge>& pub =
                            publish_completed ? state.completed : state.tree;
                        running_result.update( RunningResult( std::vector<Edge>( pub ),
                                                              DSU( state.confirmed_filter ) ) );
                    }
                    LOG_INFO( "msg", "completed prefix", "prefix", prefix );
                }
            }

            // Shut down the persistent pool: close the work channel so idle workers
            // wake and exit, then join them.
            work.close();
            for ( auto&& worker : workers ) {
                worker.join();
            }

            if ( !found.load() ) {
                throw std::runtime_error( "Minimum spanning tree not found" );
            }

            std::vector<Edge> tree( running_result.read()->tree );
            tree_weight = 0;
            for ( auto e : tree ) {
                tree_weight += e.weight;
            }
            distances_computed = count_distances;
            num_collisions = count_collisions;

            // This is just a sanity check to see if dsu works as intended
            if ( !is_connected( tree ) ) {
                throw std::runtime_error( "the returned tree is not connected" );
            }
            LOG_INFO( "msg",
                      "EMST finished",
                      "distances_computed",
                      distances_computed,
                      "num_collisions",
                      num_collisions,
                      "num_total_pairs",
                      ( (size_t)num_data - 1 ) * (size_t)num_data / 2 );
            return { tree_weight, tree };
        }

        std::pair<std::vector<Edge>, CoreDistances>
        find_tree_mutual_reachability_distance( size_t num_neighbors ) {
            clear();
            const auto find_start_t = std::chrono::steady_clock::now();

            // Collector-owned working state, constructed once and kept in sync with
            // the published snapshot across prefixes.
            MRReducerState state( num_data );
            state.core_distances =
                CoreDistances::random<Dataset, Distance>( table.get_dataset(), num_neighbors );
            // Sharpen the core-distance estimates with the clustering tree edges,
            // whose distances have already been computed.
            for ( const auto& edge : initial_tree ) {
                state.core_distances.update( edge.a, edge.b, edge.weight );
            }

            // Start the search from the clustering-based spanning tree built at
            // construction time rather than from an empty forest. The confirmed
            // filter stays empty (no edge is confirmed yet), but the heaviest
            // mutual-reachability weight of the seeded tree already bounds the
            // weight of any MST edge (cycle property), so workers can prune
            // distances right away.
            {
                std::vector<Edge> seed( initial_tree );
                update_tree( state.tree, seed, state.core_distances );
            }
            expect( state.tree.size() == num_data - 1 );

            Billboard<MRRunningResult> running_result;
            running_result.update( MRRunningResult( std::vector<Edge>( state.tree ),
                                                    DSU( state.filter ),
                                                    CoreDistances( state.core_distances ) ) );

            // The rehash schedule is computed on the mutual-reachability weights
            // of the seeded tree, since that is the scale `max_weight` prunes at.
            // The tree is sorted by mutual-reachability weight (update_tree's
            // output order), even though the stored .weight is the raw lower bound.
            std::vector<float> mr_weights;
            mr_weights.reserve( state.tree.size() );
            for ( const auto& e : state.tree ) {
                mr_weights.push_back( state.core_distances.mutual_reachability_distance( e ) );
            }
            const std::vector<float> breaks = find_breaks( mr_weights, 10.0 );

            // Pruning raw distances against a mutual-reachability bound is safe
            // because the mutual-reachability distance dominates the raw distance.
            std::atomic<float> max_weight( mr_weights.back() );
            std::atomic_size_t count_distances( 0 ), count_collisions( 0 );
            const size_t max_threads = get_worker_count( max_repetitions );
            LOG_INFO( "msg", "parallelism config",
                      "worker_threads", max_threads,
                      "max_repetitions", max_repetitions );

            std::atomic_bool found( false );

            // Persistent worker pool: spawned once and reused across every prefix.
            // Workers loop on these channels until `work` is closed.
            Channel<WorkItem> work( max_repetitions );
            Channel<MRPartial> partials( max_repetitions );
            std::vector<std::thread> workers;
            for ( size_t tid = 0; tid < max_threads; tid++ ) {
                workers.emplace_back( EMST::worker_fun_mutual_reachability,
                                      tid,
                                      std::ref( table ),
                                      std::ref( running_result ),
                                      std::ref( found ),
                                      std::ref( max_weight ),
                                      std::ref( count_distances ),
                                      std::ref( count_collisions ),
                                      std::ref( work ),
                                      std::ref( partials ) );
            }

            bool first_build = true;
            bool seeded = false;
            for ( const float distance_break : breaks ) {
                if ( found.load() ) {
                    break;
                }
                LOG_INFO( "distance-break", distance_break );
                table.builder.reset();
                table.builder.fit( table.get_dataset(),
                                   distance_break,
                                   table.num_repetitions(),
                                   delta / ( num_data - 1 ) );

                size_t initial_prefix = max_hashbits;
                if ( first_build ) {
                    // In this case the last hash value holds the smallest hash value of
                    // the previous repetition
                    table.rebuild();
                    first_build = false;
                } else {
                    table.rehash();
                    initial_prefix--;
                }

                // Once the index is first built, seed the core distances from its
                // finest-prefix neighbors before any work item is dispatched. This
                // tightens the pruning bound the workers read, so the very first
                // repetition already discards edges that cannot improve the core
                // distances instead of hoarding them.
                if ( !seeded ) {
                    seed_core_distances( state.core_distances );
                    // Re-normalize the tree under the tightened core distances
                    // (no updates to merge, just a re-sort by the new
                    // mutual-reachability weights) and republish the snapshot so
                    // workers prune against the seeded estimates from item one.
                    std::vector<Edge> no_updates;
                    update_tree( state.tree, no_updates, state.core_distances );
                    max_weight =
                        state.core_distances.mutual_reachability_distance( state.tree.back() );
                    state.filter.compress_all();
                    running_result.update(
                        MRRunningResult( std::vector<Edge>( state.tree ),
                                         DSU( state.filter ),
                                         CoreDistances( state.core_distances ) ) );
                    seeded = true;
                }

                for ( size_t prefix = initial_prefix; prefix > 0 && !found; prefix-- ) {
                    // Enqueue this prefix's repetitions for the persistent pool.
                    for ( size_t repetition = 0; repetition < max_repetitions; repetition++ ) {
                        work.send( WorkItem{ .prefix = prefix, .repetition = repetition } );
                    }

                    // Drain exactly one partial per enqueued item (even after `found`)
                    // so no item is left in the channel and all workers return to
                    // waiting -- this guarantees index quiescence.
                    size_t completed_repetitions = 0;
                    while ( completed_repetitions < max_repetitions ) {
                        std::optional<MRPartial> partial = partials.receive();
                        expect( partial.has_value() );
                        completed_repetitions++;
                        if ( found ) {
                            // discard late partials; keep draining to balance the channel
                            continue;
                        }
                        MRPartial update = std::move( *partial );
                        // clang-format off
                        LOG_DEBUG( "logger", "collector", "msg", "received update",
                                   "update-size-core-distances", update.core_distance_edges.size());
                        // clang-format on

                        // Update the owned core distances with the incoming partial, then
                        // merge into the owned tree in place.
                        for (auto & edge : update.core_distance_edges) {
                            state.core_distances.update(edge);
                        }
                        update_tree(state.tree, update.tree_edges, state.core_distances);
                        // clang-format off
                        LOG_INFO( "logger", "collector",
                                  "tree-size", state.tree.size(),
                                  "prefix", prefix,
                                  "completed-repetitions", completed_repetitions );
                        // clang-format on

                        if ( state.tree.size() == num_data - 1 ) {
                            StoppingConditionInfo stop =
                                stopping_condition( state.tree, prefix, completed_repetitions );
                            float weight_lower_bound =
                                stop.confirmed_weight +
                                stop.edges_to_confirm * stop.heaviest_confirmed_edge;
                            LOG_INFO( "weight-lower-bound", weight_lower_bound );
                            bool should_stop =
                                stop.total_weight <= ( 1 + epsilon ) * weight_lower_bound;
                            // clang-format off
                            LOG_INFO( "logger", "collector",
                                      "stop.total_weight", stop.total_weight,
                                      "stop.confirmed_weight", stop.confirmed_weight,
                                      "stop.heaviest_confirmed_edge", stop.heaviest_confirmed_edge,
                                      "stop.edges_to_confirm", stop.edges_to_confirm,
                                      "heaviest_edge", state.tree.at(num_data-2).weight,
                                      "weight_lower_bound", weight_lower_bound,
                                      "should_stop", should_stop );
                            // clang-format on
                            max_weight =
                                state.core_distances.mutual_reachability_distance( state.tree.back() );
                            LOG_INFO( "logger", "collector", "max-weight", max_weight.load() );
                            profile.push_back( ExecutionProfileElement{
                                .elapsed_ms =
                                    std::chrono::duration_cast<std::chrono::milliseconds>(
                                        std::chrono::steady_clock::now() - find_start_t )
                                        .count(),
                                .prefix = prefix,
                                .repetition = completed_repetitions,
                                .emst_confirmed_weight = stop.confirmed_weight,
                                .emst_weight_lower_bound = weight_lower_bound,
                                .emst_max_weight = max_weight,
                                .emst_max_confirmed_weight = stop.heaviest_confirmed_edge,
                                .emst_total_weight = stop.total_weight,
                                .emst_num_confirmed = num_data - 1 - stop.edges_to_confirm } );

                            // stop if we are done
                            if ( should_stop ) {
                                LOG_INFO( "msg", "tree found, signalling stop" );
                                found = true;
                            }
                            // Fill the DSU filter with just the confirmed edges
                            state.filter.reset();
                            for ( size_t idx = 0; idx < stop.confirmed_edges; idx++ ) {
                                auto edge = state.tree.at( idx );
                                state.filter.union_sets( edge.a, edge.b );
                            }
                        } else {
                            state.filter.reset();
                        }
                        // publish the new running result: copy the owned state into the
                        // billboard (the one unavoidable copy, read by the workers).
                        state.filter.compress_all();
                        running_result.update( MRRunningResult( std::vector<Edge>( state.tree ),
                                                                DSU( state.filter ),
                                                                CoreDistances( state.core_distances ) ) );
                    }
                    LOG_INFO( "msg", "completed prefix", "prefix", prefix );
                }
            }

            // Shut down the persistent pool: close the work channel so idle workers
            // wake and exit, then join them.
            work.close();
            for ( auto&& worker : workers ) {
                worker.join();
            }

            if ( !found.load() ) {
                throw std::runtime_error( "Minimum spanning tree not found" );
            }

            auto rr = running_result.read();
            std::vector<Edge> tree( rr->tree );
            CoreDistances core_distances( rr->neighborhoods);

            distances_computed = count_distances;
            num_collisions = count_collisions;
            // This is just a sanity check to see if dsu works as intended
            if ( !is_connected( tree ) ) {
                throw std::runtime_error( "the returned tree is not connected" );
            }
            LOG_INFO( "msg",
                      "EMST finished",
                      "distances_computed",
                      distances_computed,
                      "num_collisions",
                      num_collisions,
                      "num_total_pairs",
                      ( (size_t )num_data - 1 ) * (size_t)num_data / 2 );

            // Reweight the output edges with the mutual reachability
            // distance (which was already used in update_tree).
            // Furthermore, switch to the Euclidean distance, if
            // the metric used was something different
            for ( size_t i = 0; i < tree.size(); i++ ) {
                const float w = Distance::to_euclidean( tree[i].weight );
                const float ca =
                    Distance::to_euclidean( core_distances.core_distance( tree[i].a ) );
                const float cb =
                    Distance::to_euclidean( core_distances.core_distance( tree[i].b ) );
                tree[i].weight = std::max( { w, ca, cb } );
            }

            return { tree, core_distances };
        }

        //*** Private methods */
    private:

        /// Seed the core-distance estimates with genuine near neighbors read from
        /// the LSH index at its finest prefix, where buckets are smallest and thus
        /// hold the closest points. Every pair colliding at the finest prefix is
        /// fed to `CoreDistances::update`, which keeps only the `num_neighbors`
        /// smallest distances seen per point.
        ///
        /// This is always safe: `update` inserts real points at their real
        /// distances, so a point's stored k-th-neighbor distance can only move
        /// *down* toward its true value, never below it. The core distance stays a
        /// valid upper bound; seeding merely tightens it. The payoff is that the
        /// workers' `can_improve` filter starts rejecting the flood of colliding
        /// pairs immediately, instead of after each point slowly accumulates k good
        /// neighbors -- on dense datasets that flood is what exhausts memory in
        /// `possibly_useful_edges`.
        ///
        /// Requires the index to be fitted and (re)built beforehand.
        void seed_core_distances( CoreDistances& core_distances ) {
            Timer _t( "seed core distances" );
            const size_t finest_prefix = table.num_concatenations();
            const float accept_all = std::numeric_limits<float>::infinity();
            size_t seeded_pairs = 0;
            for ( size_t repetition = 0; repetition < 4; repetition++ ) {
                table.search_pairs_different_groups(
                    repetition,
                    finest_prefix,
                    1 << 16, // buffer size
                    accept_all,
                    // Every point is its own group, so no colliding pair is filtered
                    // out: we want all finest-prefix neighbors.
                    []( uint32_t x ) { return x; },
                    [&]( std::vector<Edge>& batch ) {
                        for ( const auto& e : batch ) {
                            core_distances.update( e.a, e.b, e.weight );
                        }
                        seeded_pairs += batch.size();
                        return false; // never early-stop: scan the whole prefix
                    } );
            }
            LOG_INFO( "msg", "seeded core distances", "pairs", seeded_pairs );
        }

        /// @brief Checks wheter a tree is connected
        /// @param tree the tree that we want to check
        /// @return true if all edge are connected, false otherwise.
        bool is_connected( std::vector<Edge>& tree ) {
            // Check if the tree is connected
            std::vector<Edge> edges = tree;
            std::vector<bool> visited( num_data, false );
            std::vector<std::vector<unsigned int>> adj_list( num_data );
            for ( const auto& edge : edges ) {
                adj_list[edge.a].push_back( edge.b);
                adj_list[edge.b].push_back( edge.a );
            }
            std::vector<unsigned int> stack;
            stack.push_back( 0 );
            visited.at(0) = true;
            while ( !stack.empty() ) {
                unsigned int node = stack.back();
                stack.pop_back();
                for ( const auto& neighbor : adj_list[node] ) {
                    if ( !visited.at(neighbor) ) {
                        visited.at(neighbor) = true;
                        stack.push_back( neighbor );
                    }
                }
            }
            // for (const auto& edge : tree) {
            //     std::cout << edge.first << " " << edge.second << std::endl;
            // }

            if ( !std::accumulate(
                     visited.begin(), visited.end(), true, std::logical_and<bool>() ) ) {
                LOG_INFO("msg", "Not connected");
                return false;
            }
            LOG_INFO("msg", "Connected");
            return true;
        };

        /// @brief Add the edge to the tree if it does not create a cycle using the DSU data
        /// structure
        /// @param new_edge_input the edge that we have to add
        /// @param dsu the data structure that keeps track of the connected components
        /// @param edge_list the current edges in the tree
        /// @return true if an edge has been added to the edge_list and the DSU data structure,
        /// false otherwise
        template<typename Edge>
        static bool add_edge( const Edge& new_edge, DSU& dsu, std::vector<Edge>& edge_list ) {
            // Try to add new edge normally.
            if ( dsu.union_sets( new_edge.a, new_edge.b ) ) {
                edge_list.push_back( new_edge );
                return true;
            }
            return false;
        }

        /// @brief Run Kruskal's algorithm to find the minimum spanning tree
        /// @param dsu the data structure that keeps track of the connected components
        /// @param edge_list the current edges in the tree
        /// @param output the output vector that will contain the edges in the minimum spanning tree
        template<typename Edge>
        static void kruskal( DSU& dsu, std::vector<Edge>& edge_list, std::vector<Edge>& output ) {
            for ( const auto& edge : edge_list ) {
                if ( output.size() == dsu.size() - 1 ) {
                    break;
                }
                add_edge( edge, dsu, output );
            }
        }

        /// Update the given tree with edges from the `update` list. After
        /// execution, tree will contain the minimum spanning tree
        /// on the union of `tree` and `updates`.
        /// `updates` will contain unused edges that might possibly participate
        /// in the minimum spanning tree in the future, if their mutual
        /// rechability distance lowers, as an effect of newly discovered and better
        /// neighbors
        /// `neighborhoods` is used to compute the mutual reachability distances.
        static void update_tree( std::vector<Edge>& tree,
                                 std::vector<Edge>& updates,
                                 const CoreDistances& core_distances ) {
            DSU uf( core_distances.size() );
            std::vector<MREdge> all;
            for ( auto&& e : tree ) {
                all.push_back( core_distances.mutual_reachability_edge( e ) );
            }
            for ( auto&& e : updates ) {
                all.push_back( core_distances.mutual_reachability_edge( e ) );
            }
            std::sort( all.begin(), all.end() );
            tree.clear();
            updates.clear();
            float threshold_up = -std::numeric_limits<float>::infinity();
            float threshold_low = -std::numeric_limits<float>::infinity();
            for ( auto&& e : all ) {
                if ( tree.size() == uf.size() - 1 ) {
                    break;
                }
                if ( uf.union_sets( e.a, e.b ) ) {
                    if (e.weight > threshold_up) {
                        threshold_up = e.weight;
                    }
                    if (e.lower_bound > threshold_low) {
                        threshold_low = e.lower_bound;
                    }
                    auto edge = e.as_edge();
                    expect( edge.a != edge.b );
                    expect( edge.weight >= 0 );
                    tree.push_back( e.as_edge() );
                } else {
                    // OPTIMIZE: we might be stashing some duplicates
                    updates.push_back( e.as_edge() );
                }
            }
            // expect(threshold_up >= 0);
            // expect(threshold_low >= 0);
            // auto erase_from = std::remove_if( updates.begin(), updates.end(), [&]( Edge edge ) {
            //     return !( threshold_low <= edge.weight && edge.weight <= threshold_up );
            // } );
            // updates.erase(erase_from, updates.end());
        }

        StoppingConditionInfo
        stopping_condition( const std::vector<Edge>& tree, size_t i, size_t j ) {
            const float confirmed_distance =
                table.distance_at_failure_probability( delta / ( num_data - 1 ), i, j );
            float weight = 0.0f;
            size_t idx = 0;
            while ( idx < tree.size() ) {
                const float w = tree.at(idx).weight;
                if ( w > confirmed_distance ) {
                    break;
                }
                weight += Distance::to_euclidean(w);
                idx += 1;
            }

            size_t edges_to_confirm = tree.size() - idx;

            float total_weight = weight;
            for (size_t jj=idx; jj<tree.size(); jj++) {
                float w =  tree.at(jj).weight ;
                total_weight += Distance::to_euclidean(w);
            }

            float heaviest = ( idx > 0 ) ? Distance::to_euclidean(tree.at( idx - 1 ).weight) : 0.0f;

            // All distances reported here are euclidean, so that
            // the epsilon for the approximation is applied correctly
            return StoppingConditionInfo{ .total_weight = total_weight,
                                          .confirmed_weight = weight,
                                          .heaviest_confirmed_edge = heaviest,
                                          .edges_to_confirm = edges_to_confirm,
                                          .confirmed_edges = idx };
        }

        /// @brief Clear the data structures from previous runs
        void clear() {
            distances_computed = 0;
            num_collisions = 0;
            profile.clear();
        }
    }; // closes class
} // namespace panna
