#pragma once

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <optional>
#include <queue>
#include <stdexcept>
#include <type_traits>

#include "cereal/archives/binary.hpp"
#include "panna/expect.hpp"
#include "panna/hll.hpp"
#include "panna/logging.hpp"
#include "panna/lsh/predicates.hpp"
#include "panna/prefixmap.hpp"
#include "panna/timer.hpp"

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
        public:
        typename Hasher::Builder builder;
        private:
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
            ar( repetitions,
                dataset,
                current_query,
                lsh_maps,
                builder,
                hasher,
                hashed_points );
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

        size_t memory_usage() const {
            size_t total_size = sizeof( *this );
            total_size += dataset.size() * sizeof( PointHandle );
            total_size += lsh_maps.size() * sizeof( PrefixMap<THashValue> );
            for ( const auto& map : lsh_maps ) {
                // indices is a private member, so we cannot access it directly
                total_size += map.memory_usage();
            }
            return total_size;
        }

        size_t num_concatenations() const {
            return hasher->get_concatenations();
        }

        std::string describe_family() const {
            return builder.describe();
        }

        friend bool operator==( const Index<Dataset, Hasher, Distance>& a,
                                const Index<Dataset, Hasher, Distance>& b ) {
            return a.dataset == b.dataset && a.current_query == b.current_query &&
                 a.lsh_maps == b.lsh_maps &&
                 a.hasher == b.hasher &&
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
            LOG_INFO("msg", "rebuilding index");
            if ( !hasher.has_value() ) {
                // TODO: move the fitting call outside of here. The caller of
                // rebuild should fit the builder before calling.
                builder.fit( dataset, repetitions, 0.1/dataset.size() );
                hasher = builder.build( repetitions );
            }

            std::vector<THashValue> hashes;

#pragma omp parallel for private( hashes )
            for ( size_t i = hashed_points; i < dataset.size(); i++ ) {
                auto tid = omp_get_thread_num();
                hasher->hash( dataset[i], hashes );
                for ( size_t rep = 0; rep < lsh_maps.size(); rep++ ) {
                    lsh_maps.at(rep).insert( tid, i, hashes.at(rep) );
                }
            }

#pragma omp parallel for
            for ( size_t rep = 0; rep < lsh_maps.size(); rep++ ) {
                lsh_maps.at(rep).rebuild();
            }

            hashed_points = dataset.size();
        }

        void rehash() {
            Timer _t("rehashing");
            LOG_INFO("msg", "rehashing the index");
            std::vector<std::vector<typename Hasher::Value>> old_hashes(repetitions);
            for (size_t rep=0; rep<repetitions; rep++) {
                old_hashes[rep] = lsh_maps[rep].hash_by_id();
                lsh_maps[rep].clear();
            }

            hasher = builder.build( repetitions );

            const size_t K = num_concatenations();
            std::vector<THashValue> hashes;

#pragma omp parallel for private( hashes )
            for ( size_t i = 0; i < dataset.size(); i++ ) {
                auto tid = omp_get_thread_num();
                hasher->hash( dataset[i], hashes );
                for ( size_t rep = 0; rep < lsh_maps.size(); rep++ ) {
                    lsh_maps.at(rep).insert( tid, i, hashes.at(rep) );
                }
            }

#pragma omp parallel for
            for ( size_t rep = 0; rep < lsh_maps.size(); rep++ ) {
                lsh_maps.at( rep ).overwrite_rebuild( old_hashes.at( rep ), 0, K - 1 );
            }

            hashed_points = dataset.size();
        }

    public:

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
                cursors.push_back( lsh_maps.at(rep).create_cursor( q_hashes.at(rep) ) );
            }

            // Search
            bool stop = false;
            size_t max_concat = hasher->get_concatenations();
            for ( size_t concat = max_concat; concat > 0; concat-- ) {
                if ( stop ) {
                    break;
                }
                for ( size_t rep = 0; rep < lsh_maps.size(); rep++ ) {
                    cursors.at(rep).shorten_prefix( concat );
                    for ( auto range : cursors.at(rep).get_indices() ) {
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
            float weight_filter,
            HyperLogLog& unique_pairs,
            std::function<uint32_t( uint32_t )> group_fun,
            std::function<bool( std::vector<Edge>& )> batch_output ) const {
            expect( hasher );
            size_t distance_cnt = 0;
            size_t collision_cnt = 0;
            std::vector<Edge> scratch;
            scratch.reserve(buffer_size);

            PairPrefixMapCursorGrouped<typename Hasher::Value> cursor =
                lsh_maps.at(repetition).create_pair_cursor_grouped(
                    concatenations,
                    ( concatenations < hasher->get_concatenations() )
                        ? std::optional( concatenations + 1 )
                        : std::nullopt,
                    group_fun );

            while ( true ) {
                cursor.fill_pairs_buffer( scratch, buffer_size );
                if ( scratch.size() == 0 ) {
                    // no new pairs
                    break;
                }
                // clang-format off
                LOG_DEBUG( "repetition", repetition,
                           "prefix", concatenations,
                           "num_new_pairs", scratch.size() );
                // clang-format on
                size_t write_head = 0;
                for ( size_t i = 0; i < scratch.size(); i++ ) {
                    uint32_t a_idx = scratch.at(i).a;
                    uint32_t b_idx = scratch.at(i).b;
                    if (b_idx < a_idx) {
                        // ensure that a_idx is always smaller
                        uint32_t tmp = b_idx;
                        b_idx = a_idx;
                        a_idx = tmp;
                    }

                    PointHandle a = dataset[a_idx];
                    PointHandle b = dataset[b_idx];
                    collision_cnt++;
                    float distance = Distance::compute( a, b );
                    distance_cnt++;
                    unique_pairs.add( hash_u64( pair_key( a_idx, b_idx ) ) );
                    if ( distance <= weight_filter ) {
                        scratch.at(write_head++) = {
                            .weight = distance,
                            .a = a_idx,
                            .b = b_idx
                        };
                    }
                }
                scratch.resize(write_head);
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
                lsh_maps.at(repetition).create_pair_cursor_new(
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
                    uint32_t a_idx = std::get<0>( scratch.at(i) );
                    uint32_t b_idx = std::get<1>( scratch.at(i) );
                    if (b_idx < a_idx) {
                        // ensure that a_idx is always smaller
                        uint32_t tmp = b_idx;
                        b_idx = a_idx;
                        a_idx = tmp;
                    }

                    PointHandle a = dataset[std::get<0>( scratch.at(i) )];
                    PointHandle b = dataset[std::get<1>( scratch.at(i) )];
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

        float fail_probability( float dist, size_t concat, size_t rep ) const {
            return failure_probability( *hasher, dist, concat, rep, lsh_maps.size() );
        }

        /// Returns the largest distance that attains the given failure probability
        /// at the given concatenations and repetitions.
        float distance_at_failure_probability( float delta, size_t concat, size_t rep ) const {
            expect( hasher );

            // The failure probability is monotonically non-decreasing in the distance:
            // farther pairs have a smaller collision probability and are therefore more
            // likely to be missed. We binary-search for the largest distance whose
            // failure probability does not exceed delta.
            auto fp_at = [&]( float dist ) -> float {
                return fail_probability(dist, concat, rep);
            };

            // A distance leaving the valid domain of the metric yields a non-finite
            // failure probability; we treat such distances as unacceptable so the search
            // stays within the bracket [0, valid).
            auto acceptable = [&]( float dist ) -> bool {
                const float fp = fp_at( dist );
                return std::isfinite( fp ) && fp <= delta;
            };

            // Distance zero collides with probability one, so it never fails. If even
            // that is not acceptable (e.g. delta < 0) there is nothing to return.
            float lo = 0.0f;
            if ( !acceptable( lo ) ) {
                return lo;
            }

            // Grow an upper bound by doubling until its failure probability exceeds delta
            // (or leaves the valid domain). The doubling cap keeps the loop finite.
            float hi = 1.0f;
            for ( size_t doublings = 0; doublings < 64 && acceptable( hi ); doublings++ ) {
                lo = hi;
                hi *= 2.0f;
            }
            if ( acceptable( hi ) ) {
                // Even the largest probed distance stays below delta; return it as the
                // best available lower bound.
                return hi;
            }

            // Binary search maintaining the invariant: lo is acceptable, hi is not.
            for ( size_t iter = 0; iter < 100; iter++ ) {
                const float mid = 0.5f * ( lo + hi );
                if ( mid <= lo || mid >= hi ) {
                    break; // converged to the float resolution
                }
                if ( acceptable( mid ) ) {
                    lo = mid;
                } else {
                    hi = mid;
                }
            }
            return lo;
        }

        /// Gives the earliest iteration at which the given distance would
        /// have a failure probability smaller than delta, without considering any union
        /// bound with other pairs of points. Therefore, when used in other contexts like the
        /// EMST, the actual confirmation iteration might be later.
        std::optional<std::pair<size_t, size_t>> earliest_confirming_iteration(float distance, float delta) const {

            for (size_t prefix=num_concatenations(); prefix>0; prefix--) {
                for (size_t repetition=0; repetition<num_repetitions(); repetition++) {
                    const float fp = failure_probability(*hasher, distance, prefix, repetition+1, lsh_maps.size());
                    if (fp <= delta) {
                        return std::optional(std::make_pair(prefix, repetition));
                    }
                }
            }
            return std::nullopt;
        }

        // FIXME: I don't think this belongs here, form an API standpoint
        float get_distance( size_t a, size_t b ) const {
            PointHandle x = dataset[a];
            PointHandle y = dataset[b];
            return Distance::compute( x, y );
        }
    };
} // namespace panna
