#pragma once

#include <algorithm>
#include <atomic>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <optional>
#include <queue>
#include <stdexcept>
#include <type_traits>

#include "cereal/archives/binary.hpp"
#include "panna/expect.hpp"
#include "panna/logging.hpp"
#include "panna/lsh/predicates.hpp"
#include "panna/prefixmap.hpp"

namespace panna {
    static std::atomic<size_t> g_collisions( 0 );

    template <typename Dataset, typename Hasher, typename Distance>
    class Index {
        using PointHandle = typename Dataset::PointHandle;
        using THashValue = typename Hasher::Value;

        size_t repetitions;
        // The actual data points
        Dataset dataset;
        // Contains either one or zero points to be used
        // as the current query. This is mostly for convenience,
        // since doing this way we have that the query is formatted
        // in the same way as the data.
        Dataset current_query;
        // Hash tables used by LSH.
        std::vector<PrefixMap<THashValue>> lsh_maps;
        // How to build hash functions
        typename Hasher::Builder builder;
        // How to hash the points. Initialized upon the first call to "rebuild"
        std::optional<Hasher> hasher;

        size_t hashed_points = 0;

    public:
        Index() {
        }

        Index( size_t dimensions, typename Hasher::Builder builder, size_t repetitions ):
            repetitions( repetitions ),
            dataset( dimensions ),
            current_query( dimensions ),
            builder( builder ),
            hasher(),
            hashed_points( 0 ) {

            static_assert( std::is_same<Hasher, typename Hasher::Builder::Output>::value );
            lsh_maps.resize( repetitions );
        }

        template <typename Archive>
        void serialize( Archive& ar ) {
            ar( repetitions, dataset, current_query, lsh_maps, builder, hasher, hashed_points );
        }

        size_t num_repetitions() const {
            return repetitions;
        }

        size_t num_points() const {
            return dataset.size();
        }

        const Dataset& get_dataset() const {
            return dataset;
        }

        double memory_usage() const {
            size_t total_size = sizeof( *this );
            total_size += dataset.size() * sizeof( PointHandle );
            total_size += lsh_maps.size() * sizeof( PrefixMap<THashValue> );
            for ( const auto& map : lsh_maps ) {
                // indices is a private member, so we cannot access it directly
                total_size += map.memory_usage();
            }
            // Return in Gigabytes
            return static_cast<double>(total_size) / (1024.0 * 1024.0 * 1024.0);
        }

        size_t num_concatenations() const {
            return hasher->get_concatenations();
        }

        std::string describe_family() const {
            return builder.describe();
        }

        /// Get a const reference to the hasher (must have been built via rebuild())
        const Hasher& get_hasher() const {
            expect( hasher.has_value() );
            return *hasher;
        }

        /// Return the collision probability for a given distance under the current hasher
        float collision_probability( float dist ) const {
            return hasher->collision_probability( dist );
        }

        /// Rebuild the index with a wider hash function parameter.
        /// This clears all prefix maps, resets the hasher, widens the builder,
        /// and rehashes all data points from scratch.
        void rebuild_wider( float factor ) {
            builder.widen( factor );
            hasher.reset();
            for ( auto& map : lsh_maps ) {
                map.clear();
            }
            hashed_points = 0;
            rebuild();
        }

        /// Union into the given DSU all points that share the same bucket
        /// at the given prefix level in the specified repetition's prefix map.
        void union_at_prefix( DSU& dsu, size_t repetition, uint8_t prefix ) const {
            lsh_maps[ repetition ].union_at_prefix( dsu, prefix );
        }

        /// Iterate over all buckets at the given prefix and repetition,
        /// calling callback(begin, end) for each bucket with ≥2 points.
        template <typename F>
        void for_each_bucket( size_t repetition, uint8_t prefix, F&& callback ) const {
            lsh_maps[ repetition ].for_each_bucket( prefix, std::forward<F>( callback ) );
        }

        /// Extract compressed parent arrays for all repetitions at the given prefix.
        /// Each entry is a compressed DSU parent vector: parent[i] gives the root
        /// of point i's bucket in that repetition. Within one repetition, points
        /// with the same root were all in the same hash bucket (and thus all pairs
        /// were enumerated). This avoids the transitivity bug of merging across reps.
        std::vector<std::vector<uint32_t>> extract_prefix_groups( uint8_t prefix ) const {
            std::vector<std::vector<uint32_t>> result;
            result.reserve( lsh_maps.size() );
            for ( size_t rep = 0; rep < lsh_maps.size(); rep++ ) {
                DSU dsu( dataset.size() );
                lsh_maps[ rep ].union_at_prefix( dsu, prefix );
                dsu.compress_all();
                result.push_back( std::move( dsu.parent ) );
            }
            return result;
        }

        friend bool operator==( const Index<Dataset, Hasher, Distance>& a,
                                const Index<Dataset, Hasher, Distance>& b ) {
            return a.dataset == b.dataset && a.current_query == b.current_query &&
                   a.lsh_maps == b.lsh_maps && a.hasher == b.hasher &&
                   a.hashed_points == b.hashed_points;
        }

        void save_to( std::string path ) const {
            if ( std::filesystem::exists( path ) ) {
                throw std::invalid_argument( "path already exists" );
            }

            std::ofstream os( path, std::ios::binary );
            cereal::BinaryOutputArchive ar( os );
            ar( *this );
        }

        static Index<Dataset, Hasher, Distance> load_from( std::string path ) {
            std::ifstream is( path, std::ios::binary );
            cereal::BinaryInputArchive ar( is );

            Index<Dataset, Hasher, Distance> index;
            ar( index );
            return index;
        }

        template <typename HasherBuilder, typename InputPoint>
        static Index<Dataset, Hasher, Distance> build_or_load_from( size_t dimensions,
                                                                    HasherBuilder builder,
                                                                    size_t repetitions,
                                                                    std::vector<InputPoint>& points,
                                                                    std::string path ) {
            if ( std::filesystem::exists( path ) ) {
                std::cerr << "loading from file" << std::endl;
                return load_from( path );
            } else {
                Index<Dataset, Hasher, Distance> index( dimensions, builder, repetitions );
                for ( auto p : points ) {
                    index.insert( p.begin(), p.end() );
                }
                index.rebuild();
                return index;
            }
        }

        template <typename Iter>
        void insert( Iter begin, Iter end ) {
            dataset.push_back( begin, end );
        }

        void rebuild() {
            if ( !hasher.has_value() ) {
                builder.fit( dataset );
                hasher = builder.build( repetitions );
            }

            std::vector<THashValue> hashes;

#pragma omp parallel for private( hashes )
            for ( size_t i = hashed_points; i < dataset.size(); i++ ) {
                auto tid = omp_get_thread_num();
                // auto & hashes = tl_hash_values[tid];
                hasher->hash( dataset[i], hashes );
                for ( size_t rep = 0; rep < lsh_maps.size(); rep++ ) {
                    lsh_maps[rep].insert( tid, i, hashes[rep] );
                }
            }

#pragma omp parallel for
            for ( size_t rep = 0; rep < lsh_maps.size(); rep++ ) {
                lsh_maps[rep].rebuild();
            }

            hashed_points = dataset.size();
        }

        template <typename Iter>
        void search_brute_force( Iter begin,
                                 Iter end,
                                 size_t k,
                                 std::vector<std::pair<float, uint32_t>>& output ) {
            current_query.clear();
            current_query.push_back( begin, end );

            std::priority_queue<std::pair<float, uint32_t>> top;

            PointHandle q = current_query[0];

            for ( size_t i = 0; i < dataset.size(); i++ ) {
                float dist = Distance::compute( q, dataset[i] );
                top.emplace( dist, i );
                while ( top.size() > k ) {
                    top.pop();
                }
            }

            output.clear();
            while ( top.size() > 0 ) {
                output.push_back( top.top() );
                top.pop();
            }
            std::sort( output.begin(), output.end() );
        }

        // TODO: collect statistics of the execution, including the average distance of the
        // collisions
        template <typename Iter>
        void search( Iter begin,
                     Iter end,
                     size_t k,
                     float delta,
                     std::vector<std::pair<float, uint32_t>>& output ) {
            expect( hasher );

            size_t collisions = 0;
            // Setup
            output.clear();
            current_query.clear();
            current_query.push_back( begin, end );
            PointHandle q = current_query[0];

            // FIXME: remove this allocation
            std::vector<typename Hasher::Value> q_hashes;
            hasher->hash( q, q_hashes );

            // FIXME: remove this allocation
            std::vector<PrefixMapCursor<typename Hasher::Value>> cursors;
            for ( size_t rep = 0; rep < lsh_maps.size(); rep++ ) {
                cursors.push_back( lsh_maps[rep].create_cursor( q_hashes[rep] ) );
            }

            // Search
            bool stop = false;
            size_t max_concat = hasher->get_concatenations();
            for ( size_t concat = max_concat; concat > 0; concat-- ) {
                if ( stop ) {
                    break;
                }
                for ( size_t rep = 0; rep < lsh_maps.size(); rep++ ) {
                    cursors[rep].shorten_prefix( concat );
                    for ( auto range : cursors[rep].get_indices() ) {
                        for ( const uint32_t* it = range.first; it != range.second; it++ ) {
                            PointHandle x = dataset[*it];
                            float dist = Distance::compute( q, x );
                            collisions++;
                            if ( std::find( output.begin(),
                                            output.end(),
                                            std::make_pair( dist, *it ) ) == output.end() ) {
                                output.push_back( std::make_pair( dist, *it ) );
                                std::push_heap( output.begin(), output.end() );
                                while ( output.size() > k ) {
                                    std::pop_heap( output.begin(), output.end() );
                                    output.pop_back();
                                }
                            }
                        }
                    }

                    // check stopping condition
                    if ( output.size() == k ) {
                        float topdist =
                            output.front().first; // ! We should check on the biggest element but
                                                  // the vector is a heap so it should be the first
                        float fp = failure_probability(
                            *hasher, topdist, concat, rep + 1, lsh_maps.size() );
                        if ( fp <= delta ) {
                            stop = true;
                            break;
                        }
                    }
                }
            }
            g_collisions += collisions;
        }

        //! Enumerates in the output vector the pairs colliding in the given
        //! repetition with the given number of concatenations, but only if they
        //! belong to different groups, as indicated by the group_fun function parameter
        std::pair<size_t, size_t> search_pairs_different_groups(
            size_t repetition,
            size_t concatenations,
            size_t buffer_size,
            const std::atomic<float>& weight_filter,
            std::function<uint32_t( uint32_t )> group_fun,
            std::function<bool( std::vector<Edge>& )> batch_output,
            size_t max_distances = 0,
            std::function<bool( uint32_t, uint32_t )> pair_filter = nullptr ) const {
            expect( hasher );
            size_t distance_cnt = 0;
            size_t collision_cnt = 0;
            std::vector<Edge> scratch;
            scratch.reserve(buffer_size);

            PairPrefixMapCursorGrouped<typename Hasher::Value> cursor =
                lsh_maps[repetition].create_pair_cursor_grouped(
                    concatenations,
                    ( concatenations < hasher->get_concatenations() )
                        ? std::optional( concatenations + 1 )
                        : std::nullopt,
                    group_fun );

            while ( true ) {
                // Check distance budget
                if ( max_distances > 0 && distance_cnt >= max_distances ) {
                    LOG_DEBUG( "msg", "distance budget exhausted",
                               "repetition", repetition,
                               "distance_cnt", distance_cnt );
                    break;
                }
                cursor.fill_pairs_buffer( scratch, buffer_size );
                if ( scratch.size() == 0 ) {
                    // no new pairs
                    break;
                }
                // Read the latest weight filter from the atomic
                float wf = weight_filter.load( std::memory_order_relaxed );
                LOG_DEBUG( "repetition",
                           repetition,
                           "prefix",
                           concatenations,
                           "num_new_pairs",
                           scratch.size() );
                for ( size_t i = 0; i < scratch.size(); i++ ) {
                    uint32_t a_idx = scratch[i].a;
                    uint32_t b_idx = scratch[i].b;
                    if (b_idx < a_idx) {
                        // ensure that a_idx is always smaller
                        uint32_t tmp = b_idx;
                        b_idx = a_idx;
                        a_idx = tmp;
                    }

                    // Skip pairs already seen in previous rehash rounds
                    if ( pair_filter && pair_filter( a_idx, b_idx ) ) {
                        continue;
                    }

                    PointHandle a = dataset[a_idx];
                    PointHandle b = dataset[b_idx];
                    collision_cnt++;
                    float distance = Distance::compute( a, b );
                    scratch[i].weight = distance;
                    distance_cnt++;
                    if ( distance > wf ) {
                        continue;
                    }
                    // output.emplace_back( distance, a_idx, b_idx );
                }
                if (batch_output(scratch)) {
                    // early return if the callback says so
                    break;
                }
            }
            return {distance_cnt, collision_cnt};
        }

        // Function to return all colliding couples in a given repetition and concatenation
        std::pair<size_t, size_t>
        search_pairs_filter( size_t repetition,
                             size_t concatenations,
                             std::vector<Edge>& output,
                             float weight_filter,
                             DSU& dsu_true ) {
            expect( hasher );
            size_t distance_cnt = 0;
            size_t collision_cnt = 0;
            std::vector<std::tuple<uint32_t, uint32_t, float>> scratch;
            scratch.reserve( 1 << 16 );

            PairPrefixMapCursorNew<typename Hasher::Value> cursor =
                lsh_maps[repetition].create_pair_cursor_new(
                    concatenations,
                    ( concatenations < hasher->get_concatenations() )
                        ? std::optional( concatenations + 1 )
                        : std::nullopt );

            while ( true ) {
                cursor.fill_pairs_buffer( scratch );
                if ( scratch.size() == 0 ) {
                    // no new pairs
                    break;
                }
                LOG_DEBUG( "repetition",
                           repetition,
                           "prefix",
                           concatenations,
                           "num_new_pairs",
                           scratch.size() );
                for ( size_t i = 0; i < scratch.size(); i++ ) {
                    uint32_t a_idx = std::get<0>( scratch[i] );
                    uint32_t b_idx = std::get<1>( scratch[i] );
                    if (b_idx < a_idx) {
                        // ensure that a_idx is always smaller
                        uint32_t tmp = b_idx;
                        b_idx = a_idx;
                        a_idx = tmp;
                    }

                    PointHandle a = dataset[std::get<0>( scratch[i] )];
                    PointHandle b = dataset[std::get<1>( scratch[i] )];
                    collision_cnt++;
                    if ( dsu_true.is_connected( a_idx, b_idx ) ) {
                        continue;
                    }
                    float distance = Distance::compute( a, b );
                    distance_cnt++;
                    if ( distance > weight_filter ) {
                        continue;
                    }
                    output.emplace_back( distance, a_idx, b_idx );
                }
            }
            return {distance_cnt, collision_cnt};
        }

        // Function to return all colliding couples in a given repetition and concatenation
        void search_pairs( size_t repetition,
                           size_t concatenations,
                           std::vector<std::tuple<float, std::pair<uint32_t, uint32_t>>>& output ) {
            expect( hasher );
            // Setup
            std::vector<std::pair<const uint32_t*, const uint32_t*>> scratch( 262144 ); // 65536);
            // TO DO: Find a way to create the cursors once and for all, maybe you also have to
            // store them
            PairPrefixMapCursor<typename Hasher::Value> cursor =
                lsh_maps[repetition].create_pair_cursor();
            bool keep_going = true;
            if ( concatenations != hasher->get_concatenations() ) {
                cursor.shorten_prefix( concatenations );
            }
            while ( keep_going ) {
                size_t cursor_collisions = 0;
                std::tie( cursor_collisions, keep_going ) = cursor.next( scratch );
                size_t current_size = output.size();

                // Fill the output vector and then parallel compute the distances
                for ( size_t num = 0; num < cursor_collisions; num++ ) {
                    output.emplace_back(
                        std::numeric_limits<float>::infinity(),
                        std::make_pair( *scratch[num].first,
                                        *scratch[num].second ) ); // We put a mock value?
                }

#pragma omp parallel for
                for ( size_t num = 0; num < cursor_collisions; num++ ) {
                    uint32_t x_p, y_p;
                    std::tie( x_p, y_p ) = std::get<1>( output[current_size + num] );
                    PointHandle x = dataset[x_p];
                    PointHandle y = dataset[y_p];
                    float dist = Distance::compute( y, x );
                    // If the pairs are already in the list we just have to access them so no race
                    // conditions
                    std::get<float>( output[current_size + num] ) = dist;
                }
            }

            std::sort( output.begin(), output.end() );
            // std::cout << std::get<float>(*output.begin()) << " " <<
            // std::get<float>(*(output.end()-1)) << " "; std::cout <<
            // std::get<1>(*output.begin()).first <<" "<< std::get<1>(*output.begin()).second << " "
            // << std::get<1>(*(output.end() - 1)).first << " " << std::get<1>(*(output.end() -
            // 1)).second << std::endl;
        } // End search couples

        float fail_probability( float dist, size_t concat, size_t rep ) {
            return failure_probability( *hasher, dist, concat, rep, lsh_maps.size() );
        }

        // FIXME: I don't think this belongs here, form an API standpoint
        float get_distance( size_t a, size_t b ) {
            PointHandle x = dataset[a];
            PointHandle y = dataset[b];
            return Distance::compute( x, y );
        }
    };
} // namespace panna
