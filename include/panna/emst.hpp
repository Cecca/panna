#pragma once
#include <algorithm>
#include <atomic>
#include <iostream>
#include <iterator>
#include <limits>
#include <random>
#include <unistd.h>
#include <vector>
#include <thread>

#include "panna/billboard.hpp"
#include "panna/channel.hpp"
#include "panna/dsu.hpp"
#include "panna/logging.hpp"
#include "panna/rand.hpp"
#include "panna/trieindex.hpp"
#include "panna/git_version.hpp"

namespace panna {

    struct StoppingConditionInfo {
        const float total_weight;
        const float confirmed_weight;
        const float heaviest_confirmed_edge;
        const size_t edges_to_confirm;
        const size_t confirmed_edges;
    };

    /// The tentative minimm spanning tree as it is being constructed
    /// by multiple threads
    struct RunningResult {
        std::vector<Edge> tree;
        DSU filter;

        explicit RunningResult(): tree(), filter( 0 ) {
        }
        explicit RunningResult( std::vector<Edge>&& tree, DSU&& filter ):
            tree( tree ), filter( filter ) {
        }
    };


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
        /// the maximum distance between each point and
        /// its current farthest neighbor
        std::vector<float> max_distances;

        void do_update( uint32_t src, uint32_t dst, float dist ) {
            // Given the typical small value for num_neighbors,
            // we simply proceed by a linear scan of the points.
            if ( src == dst ) {
                return;
            }
            if ( num_neighbors == 0 ) {
                return;
            }
            if ( dist < max_distances.at(src) ) {
                size_t offset = src * num_neighbors;
                // Handle special case of num_neighbors and == 1
                if ( num_neighbors == 1 ) {
                    if ( dist < neighbors.at(offset).first ) {
                        neighbors.at(offset) = { dist, dst };
                        max_distances.at(src) = dist;
                    }
                    return;
                }

                size_t ifmax = offset; // index of first maximum
                float fmax = -std::numeric_limits<float>::infinity();
                float smax = -std::numeric_limits<float>::infinity();
                for ( size_t i = offset; i < offset + num_neighbors; i++ ) {
                    if ( neighbors.at(i).first > fmax ) {
                        smax = fmax;
                        fmax = neighbors.at(i).first;
                        ifmax = i;
                    }
                }
                // replace the farthest point
                neighbors.at(ifmax) = { dist, dst };
                // update the maximum distance
                if ( dist > smax ) {
                    max_distances.at(src) = dist;
                } else {
                    max_distances.at(src) = smax;
                }
            }
        }


    public:
        explicit CoreDistances(): CoreDistances( 0, 0 ) {
        }
        explicit CoreDistances( size_t num_points, size_t num_neighbors ):
            num_points( num_points ),
            num_neighbors( num_neighbors ),
            neighbors( num_points * num_neighbors ),
            max_distances( num_points, std::numeric_limits<float>::infinity() ) {
        }

        template <typename Dataset, typename Distance>
        static CoreDistances random( const Dataset& data, size_t num_neighbors ) {
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
                self.max_distances.at(a) = farthest;
            }
            return self;
        }

        size_t size() const {
            return num_points;
        }

        std::vector<uint32_t> get_neighbors(const uint32_t v) const {
            std::vector<uint32_t> nn;
            nn.reserve(num_neighbors);
                size_t offset = v* num_neighbors;
            for ( size_t i = offset; i < offset + num_neighbors; i++ ) {
                nn.push_back(neighbors.at(i).second);
            }
            return nn;
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

        /// the distance of the farthest among the num_points
        /// neighbors we keep track of
        float core_distance( uint32_t a ) const {
            return max_distances.at(a);
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

    struct MRRunningResult {
        std::vector<Edge> tree;
        DSU filter;
        CoreDistances neighborhoods;

        explicit MRRunningResult(): tree(), filter( 0 ), neighborhoods() {
        }
        explicit MRRunningResult( std::vector<Edge>&& tree,
                                  DSU&& filter,
                                  CoreDistances&& neighborhoods ):
            tree( tree ), filter( filter ), neighborhoods( neighborhoods ) {
        }
    };

    template <typename Dataset, typename Hasher, typename Distance>
    class EMST {
        uint32_t dimensionality;
        size_t max_repetitions;
        uint32_t max_hashbits;
        Index<Dataset, Hasher, Distance> table;
        uint32_t num_data{ 0 };
        float delta{ 0.01 };
        const float epsilon{ 0.2 };
        size_t distances_computed = 0;
        size_t num_collisions = 0;

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
            table(EMST::setup_index(data_in, dimensions, repetitions)),
            num_data( data_in.size() ),
            epsilon( epsilon ),
            distances_computed( 0 ),
            num_collisions( 0 ) {
            LOG_INFO("git-version", GIT_COMMIT_HASH);

            // Get info on the index
            max_hashbits = table.num_concatenations();
            max_repetitions = table.num_repetitions();
            delta = delta_in;// / num_data;

            // Measure the size of the index
            double index_size = table.memory_usage();
            LOG_INFO("msg", "Index constructed",
                     "L", max_repetitions,
                     "K", max_hashbits,
                     "num_data", num_data,
                     "delta", delta,
                     "index_size_Gbytes", index_size);
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
            table.rebuild();

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

        /// @brief Computes the exact MST with Kruskal's algorithm in a naive way
        /// @return weight of the exact MST
        std::pair<float, std::vector<Edge>> exact_tree() {
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
            // Sort the edges
            std::sort( all_edges.begin(), all_edges.end() );
            // Create the DSU
            DSU dsu( num_data );
            float tree_weight = 0;
            std::cout << "Creating the MST" << std::endl;
            std::vector<Edge> tree;
            kruskal( dsu, all_edges, tree );
            expect(tree.size() > 0);
            LOG_INFO( "msg", "MST created",
                      "heaviest_edge",  tree.back().weight );
            for ( const auto& edge : tree ) {
                tree_weight += edge.weight ;
            }
            return {tree_weight, tree};
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
            return {tree_weight, tree};
        }
        /// the worker function in find_tree
        static void worker_fun( const size_t tid,
                                const size_t prefix,
                                const Index<Dataset, Hasher, Distance> &table,
                                // const std::vector<Edge>& tree,
                                // const DSU& filter,
                                Billboard<RunningResult> &running_result,
                                std::atomic_bool &found,
                                std::atomic<float> &max_weight,
                                std::atomic_size_t &count_distances,
                                std::atomic_size_t &count_collisions,
                                Channel<size_t> &work,
                                Channel<std::vector<Edge>> &partials ) {
            std::vector<Edge> local_tree(running_result.read()->tree);
            for ( std::optional<size_t> orepetition = work.receive(); orepetition.has_value();
                  orepetition = work.receive() ) {
                size_t repetition = *orepetition;
                LOG_INFO( "tid", tid, "repetition", repetition, "prefix", prefix, "logger", "worker" );
                if ( found ) {
                    // Return if the tree was found
                    LOG_INFO( "tid", tid, "logger", "worker", "msg", "tree found, stopping worker" );
                    return;
                }
                DSU filter = running_result.read()->filter;
                DSU dsu( filter );
                const size_t n = dsu.size();
                auto [cnt_dist, cnt_collisions] = table.search_pairs_different_groups(
                    repetition,
                    prefix,
                    std::max( (size_t)1024, n ), // smaller buffer for more frequent callbacks
                    max_weight,
                    [&]( uint32_t x ) { return filter.cfind( x ); },
                    [&]( std::vector<Edge>& scratch ) {
                        LOG_INFO( "msg", "building tree on batch", "logger", "worker", "batch_size", scratch.size() );
                        scratch.insert( scratch.end(),
                                        std::make_move_iterator( local_tree.begin() ),
                                        std::make_move_iterator( local_tree.end() ) );
                        std::sort( scratch.begin(), scratch.end() );
                        local_tree.clear();
                        dsu.reset();
                        kruskal( dsu, scratch, local_tree );
                        // Refresh filter from the latest published result.
                        // This helps skip pairs already connected in the global tree.
                        filter = running_result.read()->filter;
                        for ( auto& e : local_tree ) {
                            filter.union_sets( e.a, e.b );
                        }
                        filter.compress_all();
                        return found.load(); // early stop if the solution has been found in the meantime
                    },
                    10 * n ); // distance budget per repetition
                count_distances += cnt_dist;
                count_collisions += cnt_collisions;
                // OPTIMIZE: do not send the edges that
                // we know are already in the best tree found so far
                partials.send( std::move( local_tree ) );
            }
        }

        static void worker_fun_mutual_reachability( const size_t tid,
                                                    const size_t prefix,
                                                    const Index<Dataset, Hasher, Distance>& table,
                                                    Billboard<MRRunningResult>& running_result,
                                                    std::atomic_bool& found,
                                                    std::atomic<float>& max_weight,
                                                    std::atomic_size_t& count_distances,
                                                    std::atomic_size_t& count_collisions,
                                                    Channel<size_t>& work,
                                                    Channel<std::vector<Edge>>& partials ) {
            for ( std::optional<size_t> orepetition = work.receive(); orepetition.has_value();
                  orepetition = work.receive() ) {
                std::vector<Edge> possibly_useful_edges;
                auto rr = running_result.read();
                std::vector<Edge> local_tree( rr->tree );
                auto neighborhoods = rr->neighborhoods;
                size_t repetition = *orepetition;
                LOG_INFO(
                    "tid", tid, "repetition", repetition, "prefix", prefix, "logger", "worker" );
                // The edges we have to keep even if they are not part of the tree,
                // because they might be updated to a smaller weight in the future
                std::vector<MREdge> non_tree_edges;
                if ( found ) {
                    // Return if the tree was found
                    LOG_INFO(
                        "tid", tid, "logger", "worker", "msg", "tree found, stopping worker" );
                    return;
                }
                DSU filter = rr->filter;
                const size_t n = filter.size();
                auto [cnt_dist, cnt_collisions] = table.search_pairs_different_groups(
                    repetition,
                    prefix,
                    std::max( (size_t)1024, n ), // smaller buffer for more frequent callbacks
                    max_weight,
                    [&]( uint32_t x ) { return filter.cfind( x ); },
                    [&]( std::vector<Edge>& updates ) {
                        update_tree( local_tree, updates, neighborhoods );
                        expect( local_tree.size() > 0 );
                        // keep track of the edges that might still be useful
                        // add the potentially useful updates to the stash. Note that
                        // updates are filtered in the `update_tree` call
                        possibly_useful_edges.insert( possibly_useful_edges.end(),
                                                      std::make_move_iterator( updates.begin() ),
                                                      std::make_move_iterator( updates.end() ) );
                        // Refresh filter from the latest published result
                        filter = running_result.read()->filter;
                        for ( auto& e : local_tree ) {
                            filter.union_sets( e.a, e.b );
                        }
                        filter.compress_all();
                        // early stop if the solution has been found in the meantime
                        return found.load();
                    },
                    10 * n ); // distance budget per repetition
                count_distances += cnt_dist;
                count_collisions += cnt_collisions;
                possibly_useful_edges.insert( possibly_useful_edges.end(),
                                              std::make_move_iterator( local_tree.begin() ),
                                              std::make_move_iterator( local_tree.end() ) );
                partials.send( std::move( possibly_useful_edges ) );
            }
        }

        /// find the minimum spanning tree, using channels to handle parallelism.
        /// If all hash prefixes are exhausted without finding a valid tree,
        /// rehashes with a 2x wider scaling factor and continues.
        /// Pairs that collided at the shortest prefix in previous rounds
        /// are tracked in a DSU and skipped in subsequent rounds.
        std::pair<float, std::vector<Edge>> find_tree() {
            clear();

            Billboard<RunningResult> running_result;
            running_result.update( RunningResult( std::vector<Edge>(), DSU( num_data ) ) );

            std::atomic<float> max_weight( std::numeric_limits<float>::infinity() );
            float tree_weight = 0;
            std::atomic_size_t count_distances( 0 ), count_collisions( 0 );
            const size_t hardware_concurrency = std::thread::hardware_concurrency();
            const size_t max_threads = ( hardware_concurrency > 1 ) ? hardware_concurrency - 1 : 1;

            std::atomic_bool found( false );

            // Track previous hashers for cumulative failure probability
            std::vector<Hasher> previous_hashers;
            // Track pairs seen at the shortest prefix across rehash rounds
            DSU seen_pairs( num_data );

            const size_t MAX_REHASH_ROUNDS = 10;
            for ( size_t rehash_round = 0;
                  rehash_round < MAX_REHASH_ROUNDS && !found;
                  rehash_round++ ) {

                if ( rehash_round > 0 ) {
                    LOG_INFO( "msg", "Rehashing with wider scaling factor",
                              "round", rehash_round );

                    // Record pairs sharing the shortest prefix (prefix 1)
                    // in the current hash tables before they are destroyed.
                    for ( size_t rep = 0; rep < max_repetitions; rep++ ) {
                        table.union_at_prefix( seen_pairs, rep, 1 );
                    }
                    seen_pairs.compress_all();

                    // Save current hasher for failure probability computation
                    previous_hashers.push_back( table.get_hasher() );

                    // Rebuild with 2x wider scaling/quantization
                    table.rebuild_wider( 2.0 );

                    // Merge seen_pairs into the filter of the running result
                    auto rr = running_result.read();
                    DSU combined_filter( num_data );
                    for ( auto& edge : rr->tree ) {
                        combined_filter.union_sets( edge.a, edge.b );
                    }
                    for ( size_t i = 0; i < num_data; i++ ) {
                        uint32_t sp_root = seen_pairs.cfind( i );
                        if ( sp_root != i ) {
                            combined_filter.union_sets( i, sp_root );
                        }
                    }
                    combined_filter.compress_all();
                    running_result.update( RunningResult(
                        std::vector<Edge>( rr->tree ), std::move( combined_filter ) ) );
                }

                for ( size_t prefix = max_hashbits; prefix > 0 && !found; prefix-- ) {
                    // Set up work to distribute among threads
                    Channel<size_t> work( max_repetitions );
                    for ( size_t repetition = 0; repetition < max_repetitions; repetition++ ) {
                        work.send( std::move( repetition ) );
                    }
                    work.close();

                    // Set up the channel to collect partial results
                    Channel<std::vector<Edge>> partials( max_repetitions );

                    // Spawn worker threads
                    std::vector<std::thread> workers;
                    for ( size_t tid = 0; tid < max_threads; tid++ ) {
                        std::thread worker( EMST::worker_fun,
                                            tid,
                                            prefix,
                                            std::ref(table),
                                            std::ref(running_result),
                                            std::ref(found),
                                            std::ref(max_weight),
                                            std::ref(count_distances),
                                            std::ref(count_collisions),
                                            std::ref(work),
                                            std::ref(partials) );
                        workers.push_back( std::move( worker ) );
                    }

                    // Collect results from workers, batching for efficiency
                    size_t completed_repetitions = 0;
                    std::vector<Edge> batch_updates;
                    const size_t BATCH_SIZE = std::max( (size_t)1, max_threads / 2 );
                    size_t batch_count = 0;

                    for ( std::optional<std::vector<Edge>> local_tree = partials.receive();
                          local_tree.has_value() && !found && completed_repetitions < max_repetitions;
                          local_tree = partials.receive() ) {
                        std::vector<Edge> update = std::move( *local_tree );
                        LOG_INFO( "logger", "collector", "msg", "received update",
                                  "update-size", update.size() );

                        completed_repetitions++;
                        batch_count++;

                        // Accumulate partial results into a batch
                        batch_updates.insert( batch_updates.end(),
                            std::make_move_iterator( update.begin() ),
                            std::make_move_iterator( update.end() ) );

                        // Process batch when full or when all repetitions are done
                        bool should_process_batch =
                            ( batch_count >= BATCH_SIZE ) ||
                            ( completed_repetitions >= max_repetitions );
                        if ( !should_process_batch ) continue;
                        batch_count = 0;

                        // Merge: the running result tree is sorted, so we can use merge
                        std::vector<Edge> tree( running_result.read()->tree );
                        DSU filter( num_data );
                        std::sort( batch_updates.begin(), batch_updates.end() );
                        std::vector<Edge> merged;
                        merged.reserve( tree.size() + batch_updates.size() );
                        std::merge( tree.begin(), tree.end(),
                                    batch_updates.begin(), batch_updates.end(),
                                    std::back_inserter( merged ) );
                        tree.clear();
                        kruskal( filter, merged, tree );
                        batch_updates.clear();
                        merged.clear();

                        LOG_INFO( "logger", "collector", "tree-size", tree.size(),
                                  "completed-repetitions", completed_repetitions );

                        if ( tree.size() == num_data - 1 ) {
                            StoppingConditionInfo stop =
                                stopping_condition_with_history(
                                    tree, prefix, completed_repetitions,
                                    previous_hashers );
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
                                      "heaviest_edge", tree.at(num_data-2).weight,
                                      "weight_lower_bound", weight_lower_bound,
                                      "should_stop", should_stop,
                                      "rehash_round", rehash_round );
                            // clang-format on
                            max_weight = tree.back().weight;
                            LOG_INFO( "logger", "collector", "max-weight", max_weight.load() );

                            if ( should_stop ) {
                                LOG_INFO( "msg", "tree found, signalling stop" );
                                found = true;
                                tree_weight = stop.total_weight;
                            }
                            // Fill the DSU filter with confirmed edges + seen pairs
                            filter.reset();
                            for ( size_t idx = 0; idx < stop.confirmed_edges; idx++ ) {
                                auto edge = tree.at(idx);
                                filter.union_sets( edge.a, edge.b );
                            }
                            // Merge seen_pairs into filter so workers skip them
                            for ( size_t i = 0; i < num_data; i++ ) {
                                uint32_t sp_root = seen_pairs.cfind( i );
                                if ( sp_root != i ) {
                                    filter.union_sets( i, sp_root );
                                }
                            }
                        } else {
                            filter.reset();
                            // Still merge seen_pairs when tree is incomplete
                            for ( size_t i = 0; i < num_data; i++ ) {
                                uint32_t sp_root = seen_pairs.cfind( i );
                                if ( sp_root != i ) {
                                    filter.union_sets( i, sp_root );
                                }
                            }
                        }
                        // Publish the new running result
                        filter.compress_all();
                        running_result.update(
                            RunningResult( std::move( tree ), std::move( filter ) ) );

                        if ( completed_repetitions >= max_repetitions ) {
                            break;
                        }
                    }

                    // Wait for workers to finish
                    for ( auto&& worker : workers ) {
                        worker.join();
                    }
                    LOG_INFO( "msg", "completed prefix", "prefix", prefix );
                }

                LOG_INFO( "msg", "completed rehash round", "round", rehash_round );
            }

            std::vector<Edge> tree(running_result.read()->tree);
            tree_weight = 0;
            for (auto e : tree) {
                tree_weight += e.weight;
            }
            distances_computed = count_distances;
            num_collisions = count_collisions;

            // This is just a sanity check to see if dsu works as intended
            is_connected( tree );
            LOG_INFO( "msg", "EMST finished",
                      "distances_computed", distances_computed,
                      "num_collisions", num_collisions,
                      "num_total_pairs", ((size_t)num_data -1) *(size_t) num_data/ 2 );
            return { tree_weight, tree };
        }

        std::pair<std::vector<Edge>, CoreDistances>
        find_tree_mutual_reachability_distance( size_t num_neighbors ) {
            clear();

            CoreDistances cs =
                CoreDistances::random<Dataset, Distance>( table.get_dataset(), num_neighbors );
            Billboard<MRRunningResult> running_result;
            running_result.update(
                MRRunningResult( std::vector<Edge>(), DSU( num_data ), std::move( cs ) ) );

            std::atomic<float> max_weight( std::numeric_limits<float>::infinity() );
            std::atomic_size_t count_distances( 0 ), count_collisions( 0 );
            const size_t hardware_concurrency = std::thread::hardware_concurrency();
            const size_t max_threads = ( hardware_concurrency > 1 ) ? hardware_concurrency - 1 : 1;

            std::atomic_bool found( false );
            for ( size_t prefix = max_hashbits; prefix > 0 && !found; prefix-- ) {
                // Set up work to distribute among threads: each worker thread will pull
                // repetition indices from this
                Channel<size_t> work( max_repetitions );
                for ( size_t repetition = 0; repetition < max_repetitions; repetition++ ) {
                    work.send( std::move( repetition ) );
                }
                // Close the channel, so that workers do not wait indefinitely for new repetitions
                work.close();

                // Set up the channel to collect partial results
                Channel<std::vector<Edge>> partials( max_repetitions );

                // spawn the threads to carry out the work
                std::vector<std::thread> workers;
                for ( size_t tid = 0; tid < max_threads; tid++ ) {
                    std::thread worker( EMST::worker_fun_mutual_reachability,
                                        tid,
                                        prefix,
                                        std::ref( table ),
                                        std::ref( running_result ),
                                        std::ref( found ),
                                        std::ref( max_weight ),
                                        std::ref( count_distances ),
                                        std::ref( count_collisions ),
                                        std::ref( work ),
                                        std::ref( partials ) );
                    workers.push_back( std::move( worker ) );
                }

                // collect the results from the worker threads
                size_t completed_repetitions = 0;
                std::vector<Edge> stash;
                for ( std::optional<std::vector<Edge>> local_tree = partials.receive();
                      local_tree.has_value() && !found && completed_repetitions < max_repetitions;
                      local_tree = partials.receive() ) {
                    DSU filter(num_data);
                    std::vector<Edge> update = std::move( *local_tree );
                    // clang-format off
                    LOG_INFO( "logger", "collector", "msg", "received update", "update-size", update.size(), "stash-size", stash.size());
                    // clang-format: on
                    update.insert(update.end(), stash.begin(), stash.end());

                    completed_repetitions++;

                    std::vector<Edge> tree( running_result.read()->tree );
                    CoreDistances core_distances(running_result.read()->neighborhoods);
                    for (auto & edge : update) {
                        core_distances.update(edge);
                    }
                    update_tree(tree, update, core_distances);
                    // stash the edges that might be useful in the future
                    stash = std::move(update);
                    // clang-format off
                    LOG_INFO( "logger", "collector", "tree-size", tree.size(), "completed-repetitions", completed_repetitions , "stash-size", stash.size());
                    // clang-format on

                    if ( tree.size() == num_data - 1 ) {
                        StoppingConditionInfo stop =
                            stopping_condition( tree, prefix, completed_repetitions );
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
                                  "heaviest_edge", tree.at(num_data-2).weight,
                                  "weight_lower_bound", weight_lower_bound,
                                  "should_stop", should_stop );
                        // clang-format on
                        max_weight = core_distances.mutual_reachability_distance(tree.back());
                        LOG_INFO( "logger", "collector", "max-weight", max_weight.load() );

                        // stop if we are done
                        if ( should_stop ) {
                            LOG_INFO( "msg", "tree found, signalling stop" );
                            found = true;
                        }
                        // Fill the DSU filter with just the confirmed edges
                        filter.reset();
                        for ( size_t idx = 0; idx < stop.confirmed_edges; idx++ ) {
                            auto edge = tree.at(idx);
                            filter.union_sets( edge.a, edge.b );
                        }
                    } else {
                        filter.reset();
                    }
                    // publish the new running result
                    filter.compress_all();
                    running_result.update( MRRunningResult(
                        std::move( tree ), std::move( filter ), std::move( core_distances ) ) );

                    if ( completed_repetitions >= max_repetitions ) {
                        // we are done with this prefix
                        break;
                    }
                }

                // Wait for workers to finish
                for ( auto&& worker : workers ) {
                    worker.join();
                }
                LOG_INFO( "msg", "completed prefix", "prefix", prefix );
            }

            auto rr = running_result.read();
            std::vector<Edge> tree( rr->tree );
            CoreDistances core_distances( rr->neighborhoods);

            distances_computed = count_distances;
            num_collisions = count_collisions;
            // This is just a sanity check to see if dsu works as intended
            is_connected( tree );
            LOG_INFO( "msg",
                      "EMST finished",
                      "distances_computed",
                      distances_computed,
                      "num_collisions",
                      num_collisions,
                      "num_total_pairs",
                      ( (size_t )num_data - 1 ) * (size_t)num_data / 2 );
            return { tree, core_distances };
        }

        //*** Private methods */
    private:

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
                    tree.push_back( e.as_edge() );
                } else {
                    // FIXME: we might be stashing some duplicates
                    updates.push_back( e.as_edge() );
                }
            }
            expect(threshold_up >= 0);
            expect(threshold_low >= 0);
            auto erase_from = std::remove_if( updates.begin(), updates.end(), [&]( Edge edge ) {
                return !( threshold_low <= edge.weight && edge.weight <= threshold_up );
            } );
            updates.erase(erase_from, updates.end());
        }

        /// @brief Generate a random spanning tree to have an initial solution
        void dirty_start( std::vector<Edge>& clean ) {
            std::vector<unsigned int> vertices( num_data );
            std::iota( vertices.begin(), vertices.end(), 0 );
            std::shuffle(
                vertices.begin(), vertices.end(), std::mt19937{ std::random_device{}() } );
            for ( size_t i = 1; i < vertices.size(); i++ ) {
                clean.emplace_back( table.get_distance( vertices.at(i - 1), vertices.at(i) ),
                                    std::make_pair( vertices.at(i - 1), vertices.at(i) ) );
            }
        }

        StoppingConditionInfo stopping_condition( std::vector<Edge> tree, size_t i, size_t j ) {
            std::vector<Hasher> empty;
            return stopping_condition_with_history( tree, i, j, empty );
        }

        /// Like stopping_condition, but accounts for cumulative failure probability
        /// across multiple rehashing rounds. Each entry in previous_hashers
        /// represents a completed round where all L repetitions were exhausted
        /// at all prefix levels.
        StoppingConditionInfo stopping_condition_with_history(
            std::vector<Edge>& tree,
            size_t i,
            size_t j,
            const std::vector<Hasher>& previous_hashers ) {
            float prob = 0.0f;
            float weight = 0.0f;
            size_t idx = 0;
            while ( idx < tree.size() ) {
                const float w = tree.at(idx).weight;

                // Current round failure probability
                float fp = table.fail_probability( w, i, j );

                // Multiply by each previous round's failure probability.
                // A completed round at prefix 1 with all L repetitions
                // has failure prob = (1 - cp)^L
                for ( const auto& prev_h : previous_hashers ) {
                    float cp = prev_h.collision_probability( w );
                    fp *= std::pow( 1.0f - cp,
                                    static_cast<float>( max_repetitions ) );
                }

                if ( prob + fp > delta ) {
                    break;
                }
                prob += fp;
                weight += w;
                idx += 1;
            }

            size_t edges_to_confirm = tree.size() - idx;

            float total_weight = weight;
            for (size_t jj=idx; jj<tree.size(); jj++) {
                float w =  tree.at(jj).weight ;
                total_weight += w;
            }

            return StoppingConditionInfo{ .total_weight = total_weight,
                                          .confirmed_weight = weight,
                                          .heaviest_confirmed_edge =
                                              ( idx > 0 ) ?  tree.at(idx - 1).weight  : 0.0f,
                                          .edges_to_confirm = edges_to_confirm,
                                          .confirmed_edges = idx };
        }

        /// @brief Clear the data structures from previous runs
        void clear() {
            distances_computed = 0;
            num_collisions = 0;
        }
    }; // closes class
} // namespace panna
