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

        // Stores per-point bucket ids using the minimum fixed-width integer
        // required by a specific rehash cycle.
        struct CompactBucketIds {
            uint8_t bytes_per_id = 0;
            std::vector<uint8_t> raw;

            template <typename Archive>
            void serialize( Archive& ar ) {
                ar( bytes_per_id, raw );
            }

            size_t size() const {
                if ( bytes_per_id == 0 ) {
                    return 0;
                }
                return raw.size() / bytes_per_id;
            }

            uint32_t at( uint32_t idx ) const {
                expect( bytes_per_id == 1 || bytes_per_id == 2 || bytes_per_id == 4 );
                const size_t offset = static_cast<size_t>( idx ) * bytes_per_id;
                expect( offset + bytes_per_id <= raw.size() );
                if ( bytes_per_id == 1 ) {
                    return raw.at( offset );
                }
                if ( bytes_per_id == 2 ) {
                    uint16_t value;
                    std::memcpy( &value, raw.data() + offset, sizeof( value ) );
                    return value;
                }
                uint32_t value;
                std::memcpy( &value, raw.data() + offset, sizeof( value ) );
                return value;
            }

            static CompactBucketIds from_dense( const std::vector<uint32_t>& dense ) {
                CompactBucketIds compact;
                if ( dense.empty() ) {
                    return compact;
                }

                const uint32_t max_bucket = *std::max_element( dense.begin(), dense.end() );
                if ( max_bucket <= std::numeric_limits<uint8_t>::max() ) {
                    compact.bytes_per_id = 1;
                } else if ( max_bucket <= std::numeric_limits<uint16_t>::max() ) {
                    compact.bytes_per_id = 2;
                } else {
                    compact.bytes_per_id = 4;
                }

                compact.raw.resize( dense.size() * compact.bytes_per_id );
                for ( size_t i = 0; i < dense.size(); i++ ) {
                    const size_t offset = i * compact.bytes_per_id;
                    if ( compact.bytes_per_id == 1 ) {
                        compact.raw.at( offset ) = static_cast<uint8_t>( dense.at( i ) );
                    } else if ( compact.bytes_per_id == 2 ) {
                        const uint16_t value = static_cast<uint16_t>( dense.at( i ) );
                        std::memcpy( compact.raw.data() + offset, &value, sizeof( value ) );
                    } else {
                        const uint32_t value = dense.at( i );
                        std::memcpy( compact.raw.data() + offset, &value, sizeof( value ) );
                    }
                }
                return compact;
            }

            friend bool operator==( const CompactBucketIds& a, const CompactBucketIds& b ) {
                return a.bytes_per_id == b.bytes_per_id && a.raw == b.raw;
            }
        };

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
        // Stores prefix-1 bucket ids from previous rehashes in cycle-major order:
        // index = cycle * num_repetitions + repetition.
        std::vector<CompactBucketIds> rehash_prefix_buckets;
        size_t rehash_cycles = 0;
        // Hashers used in completed rehash cycles, needed to account for
        // different quantization widths in failure probability estimates.
        std::vector<Hasher> rehash_history_hashers;
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
            rehash_prefix_buckets.clear();
            rehash_prefix_buckets.reserve( repetitions );
        }

        template <typename Archive>
        void serialize( Archive& ar ) {
            ar( repetitions,
                dataset,
                current_query,
                lsh_maps,
                rehash_prefix_buckets,
                rehash_cycles,
                rehash_history_hashers,
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
                 a.rehash_prefix_buckets == b.rehash_prefix_buckets &&
                 a.rehash_cycles == b.rehash_cycles &&
                 a.rehash_history_hashers == b.rehash_history_hashers &&
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
                builder.fit( dataset );
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

    private:
        void record_current_prefix_buckets() {
            if ( lsh_maps.empty() ) {
                return;
            }
            expect( hasher.has_value() );
            rehash_history_hashers.push_back( *hasher );
            const size_t reps = lsh_maps.size();
            rehash_prefix_buckets.reserve( rehash_prefix_buckets.size() + reps );
            for ( size_t rep = 0; rep < lsh_maps.size(); rep++ ) {
                std::vector<uint32_t> buckets;
                buckets.reserve( dataset.size() );
                lsh_maps.at(rep).fill_prefix_bucket_ids( 1, buckets );
                rehash_prefix_buckets.push_back( CompactBucketIds::from_dense( buckets ) );
            }
            rehash_cycles++;
        }

        bool seen_in_previous_rehashes( size_t repetition, uint32_t a_idx, uint32_t b_idx ) const {
            const size_t reps = lsh_maps.size();
            if ( repetition >= reps ) {
                return false;
            }
            for ( size_t cycle = 0; cycle < rehash_cycles; cycle++ ) {
                const auto& buckets =
                    rehash_prefix_buckets.at( cycle * reps + repetition );
                if ( a_idx < buckets.size() && b_idx < buckets.size() &&
                     buckets.at(a_idx) == buckets.at(b_idx) ) {
                    return true;
                }
            }
            return false;
        }

    public:

        /// Rehash the index so that at the longest prefix there is a reasonable number of
        /// collisions of points from different groups.
        void rehash( std::function<uint32_t( uint32_t )> group_fun ) {
            Timer timer("reshashing");
            if ( hasher.has_value() ) {
                record_current_prefix_buckets();
            }
            builder.fit(dataset, group_fun);
            hasher = builder.build( repetitions );
            for (auto & map : lsh_maps) {
                map.clear();
            }
            hashed_points = 0;
            rebuild();
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

                    if ( seen_in_previous_rehashes( repetition, a_idx, b_idx ) ) {
                        continue;
                    }

                    PointHandle a = dataset[a_idx];
                    PointHandle b = dataset[b_idx];
                    collision_cnt++;
                    float distance = Distance::compute( a, b );
                    distance_cnt++;
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

                    if ( seen_in_previous_rehashes( repetition, a_idx, b_idx ) ) {
                        continue;
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
                lsh_maps.at(repetition).create_pair_cursor();
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
                        std::make_pair( *scratch.at(num).first,
                                        *scratch.at(num).second ) ); // We put a mock value?
                }

#pragma omp parallel for
                for ( size_t num = 0; num < cursor_collisions; num++ ) {
                    uint32_t x_p, y_p;
                    std::tie( x_p, y_p ) = std::get<1>( output.at(current_size + num) );
                    PointHandle x = dataset[x_p];
                    PointHandle y = dataset[y_p];
                    float dist = Distance::compute( y, x );
                    // If the pairs are already in the list we just have to access them so no race
                    // conditions
                    std::get<float>( output.at(current_size + num) ) = dist;
                }
            }

            std::sort( output.begin(), output.end() );
            // std::cout << std::get<float>(*output.begin()) << " " <<
            // std::get<float>(*(output.end()-1)) << " "; std::cout <<
            // std::get<1>(*output.begin()).first <<" "<< std::get<1>(*output.begin()).second << " "
            // << std::get<1>(*(output.end() - 1)).first << " " << std::get<1>(*(output.end() -
            // 1)).second << std::endl;
        } // End search couples

        float fail_probability( float dist, size_t concat, size_t rep ) const {
            const float current_cycle_failure =
                failure_probability( *hasher, dist, concat, rep, lsh_maps.size() );

            // Each completed rehash cycle fully exhausts prefix=1 for all repetitions,
            // but with its own quantization width (different hasher fit). We therefore
            // multiply residual failures using the exact historical hasher of each cycle.
            float history_failure = 1.0f;
            for ( const auto& historical_hasher : rehash_history_hashers ) {
                const size_t historical_reps = historical_hasher.get_repetitions();
                history_failure *= failure_probability(
                    historical_hasher,
                    dist,
                    1,
                    historical_reps,
                    historical_reps );
            }

            return current_cycle_failure * history_failure;
        }

        /// Returns the largest distance that attains the given failure probability
        /// at the given concatenations and repetitions.
        float distance_at_failure_probability( float delta, size_t concat, size_t rep ) const {
            expect( hasher );

            // The failure probability is monotonically non-decreasing in the distance:
            // farther pairs have a smaller collision probability and are therefore more
            // likely to be missed. We binary-search for the largest distance whose
            // failure probability (combined across all rehash cycles, exactly as in
            // fail_probability) does not exceed delta.
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
