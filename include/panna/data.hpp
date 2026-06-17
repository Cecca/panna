#pragma once

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <omp.h>
#include <random>
#include <stdexcept>
#include <vector>

#include "panna/expect.hpp"
#include "panna/logging.hpp"
#include "panna/rand.hpp"

namespace panna {
    //! A chunk of a longer vector of 16bit integers.
    //! On C++17 this class is guaranteed to be aligned to the 256-bit boundary
    //! even when stored in a std::vector.
    struct alignas( 32 ) Int16Chunk {
        static constexpr size_t CHUNK_SIZE = 16;
        // int16_t chunk[CHUNK_SIZE];
        std::array<int16_t, CHUNK_SIZE> chunk;

        template <typename Archive>
        void serialize( Archive& ar ) {
            ar( chunk );
        }

        friend bool operator==( const Int16Chunk& a, const Int16Chunk& b ) {
            return a.chunk == b.chunk;
        }
    };
    static_assert( sizeof( Int16Chunk ) == 256 / 8, "int16 chunk should fill 256 bits" );
    static_assert( alignof( Int16Chunk ) == 256 / 8, "int16 chunk should align to 256 bits" );

    static constexpr int16_t to_16bit_fixed_point( float val ) {
        assert( val >= -1.0 && val <= 1.0 );

        val = std::min( val * ( 1 << 15 ), static_cast<float>( INT16_MAX ) );
        return static_cast<int16_t>( val );
    }

    static constexpr float from_16bit_fixed_point( int16_t val ) {
        return static_cast<float>( val ) / ( 1 << 15 );
    }

    struct UnitNormPointHandle {
        Int16Chunk const* chunks;
        size_t num_chunks;
        size_t dimensions;

        void into_vec( std::vector<float>& vec ) const {
            // in some cases (like cross polytope) the output vector holds more
            // elements than the dimensions
            expect( vec.size() >= dimensions );
            size_t i = 0;
            for ( size_t chunk_idx = 0; chunk_idx < num_chunks; chunk_idx++ ) {
                Int16Chunk chunk = chunks[chunk_idx];
                for ( size_t j = 0; j < Int16Chunk::CHUNK_SIZE; j++ ) {
                    if ( i >= dimensions ) {
                        break;
                    }
                    vec.at(i) = from_16bit_fixed_point( chunk.chunk[j] );
                    i++;
                }
            }
        }

        friend std::ostream& operator<<( std::ostream& os, const UnitNormPointHandle& handle ) {
            std::vector<float> vec( handle.dimensions );
            handle.into_vec( vec );
            os << "[";
            for ( auto x : vec ) {
                os << x << ", ";
            }
            os << "]";
            return os;
        }
    };

    class UnitNormPoints {

        size_t dimensions = 0;
        size_t padding = 0;
        size_t chunks_per_point = 0;

        // Since C++17 the allocation of std::vector respects the alignment of
        // the template argument.
        std::vector<Int16Chunk> chunks;

    public:
        using PointHandle = UnitNormPointHandle;

        UnitNormPoints() {
        }

        UnitNormPoints( size_t dimensions ):
            dimensions( dimensions ),
            padding( ( Int16Chunk::CHUNK_SIZE - ( dimensions % Int16Chunk::CHUNK_SIZE ) ) %
                     Int16Chunk::CHUNK_SIZE ),
            chunks_per_point( std::ceil( ( (float)dimensions ) / Int16Chunk::CHUNK_SIZE ) ) {
        }

        template <typename Archive>
        void serialize( Archive& ar ) {
            ar( dimensions, padding, chunks_per_point, chunks );
        }

        friend bool operator==( const UnitNormPoints& a, const UnitNormPoints& b ) {
            return a.dimensions == b.dimensions && a.padding == b.padding && a.chunks_per_point &&
                   b.chunks_per_point && a.chunks == b.chunks;
        }

        void clear() {
            chunks.clear();
        }

        size_t get_padding() const {
            return padding;
        }

        size_t get_chunks_per_point() const {
            return chunks_per_point;
        }

        PointHandle operator[]( size_t i ) const {
            assert( i * chunks_per_point < chunks.size() );
            UnitNormPointHandle handle;
            handle.chunks = &chunks.at(i * chunks_per_point);
            handle.num_chunks = chunks_per_point;
            handle.dimensions = dimensions;
            return handle;
        }

        template <typename FloatIter>
        void push_back( FloatIter begin, FloatIter end ) {
            float sq_norm = 0.0;
            for ( FloatIter it = begin; it != end; it++ ) {
                float v = *it;
                sq_norm += v * v;
            }

            // prepare the space
            size_t base_chunk_idx = chunks.size();
            for ( size_t i = 0; i < chunks_per_point; i++ ) {
                chunks.emplace_back();
            }

            auto norm = std::sqrt( sq_norm );
            size_t i = 0;
            for ( FloatIter it = begin; it != end; it++ ) {
                float normalized = ( norm == 0.0 ) ? *it : *it / norm;
                chunks.at(base_chunk_idx + i / Int16Chunk::CHUNK_SIZE)
                    .chunk[i % Int16Chunk::CHUNK_SIZE] = to_16bit_fixed_point( normalized );
                i++;
            }
            for ( size_t i = dimensions; i < dimensions + padding; i++ ) {
                chunks.at(base_chunk_idx + i / Int16Chunk::CHUNK_SIZE)
                    .chunk[i % Int16Chunk::CHUNK_SIZE] = to_16bit_fixed_point( 0.0 );
            }
        }

        PointHandle push_back_random() {
            std::vector<float> values;
            for ( unsigned int i = 0; i < dimensions; i++ ) {
                values.push_back( sample_random_normal() );
            }

            push_back( values.begin(), values.end() );
            return operator[]( size() - 1 );
        }

        size_t size() const {
            return chunks.size() / chunks_per_point;
        }

        size_t get_dimensions() const {
            return dimensions;
        }
    };

    struct NormedPointHandle {
        UnitNormPointHandle inner;
        float sq_norm;

        float squared_norm() const {
            return sq_norm;
        }

        void into_vec( std::vector<float>& vec ) const {
            // in some cases (like cross polytope) the output vector holds more
            // elements than the dimensions
            expect( vec.size() >= inner.dimensions );
            inner.into_vec(vec);
            const float norm = std::sqrt( sq_norm );
            for (size_t i=0; i<inner.dimensions; i++) {
                vec.at(i) *= norm;
            }
        }

        friend std::ostream& operator<<( std::ostream& os, const NormedPointHandle& handle ) {
            std::vector<float> vec( handle.inner.dimensions );
            handle.inner.into_vec( vec );
            float norm = std::sqrt( handle.sq_norm );
            os << "[";
            for ( auto x : vec ) {
                os << norm * x << ", ";
            }
            os << "]";
            return os;
        }
    };

    class NormedPoints {
        size_t dimensions = 0;
        UnitNormPoints normalized_points;
        std::vector<float> squared_norms;

    public:
        using PointHandle = NormedPointHandle;

        NormedPoints() {
        }

        NormedPoints( size_t dimensions ):
            dimensions( dimensions ), normalized_points( dimensions ) {
        }

        template <typename Archive>
        void serialize( Archive& ar ) {
            ar( dimensions, normalized_points, squared_norms );
        }

        friend bool operator==( const NormedPoints& a, const NormedPoints& b ) {
            return a.dimensions == b.dimensions && a.normalized_points == b.normalized_points &&
                   a.squared_norms == b.squared_norms;
        }

        void clear() {
            normalized_points.clear();
            squared_norms.clear();
        }

        PointHandle operator[]( size_t i ) const {
            PointHandle handle;
            handle.inner = normalized_points[i];
            handle.sq_norm = squared_norms.at(i);
            return handle;
        }

        template <typename FloatIter>
        void push_back( FloatIter begin, FloatIter end ) {
            float sq_norm = 0.0;
            for ( FloatIter it = begin; it != end; it++ ) {
                float v = *it;
                sq_norm += v * v;
            }
            normalized_points.push_back( begin, end );
            squared_norms.push_back( sq_norm );
        }

        PointHandle push_back_random() {
            std::vector<float> values;
            for ( unsigned int i = 0; i < dimensions; i++ ) {
                values.push_back( sample_random_normal() );
            }

            push_back( values.begin(), values.end() );
            return operator[]( size() - 1 );
        }

        size_t size() const {
            return squared_norms.size();
        }
        
        size_t get_dimensions() const {
            return dimensions;
        }
    };

    struct EuclideanPointHandle {
        const size_t dimensions;
        const float * vector;

        void into_vec( std::vector<float>& vec ) const {
            // in some cases (like cross polytope) the output vector holds more
            // elements than the dimensions
            expect( vec.size() >= dimensions );
            for (size_t i=0; i<dimensions; i++) {
                vec.at(i) = vector[i];
            }
        }

        friend std::ostream& operator<<( std::ostream& os, const EuclideanPointHandle& handle ) {
            std::vector<float> vec( handle.dimensions );
            os << "[";
            for (size_t i=0; i<handle.dimensions; i++) {
                os << handle.vector[i] << ", ";
            }
            os << "]";
            return os;
        }
    };

    class EuclideanPoints {
        size_t dimensions = 0;
        // OPTIMIZE: use a better layout, with padding and alignment,
        // so to enable SIMD optimization
        std::vector<float> data;

    public:
        using PointHandle = EuclideanPointHandle;

        EuclideanPoints() {
        }

        EuclideanPoints( size_t dimensions ):
            dimensions( dimensions ), data( ) {
        }

        template <typename Archive>
        void serialize( Archive& ar ) {
            ar( dimensions, data );
        }

        friend bool operator==( const EuclideanPoints& a, const EuclideanPoints& b ) {
            return a.dimensions == b.dimensions && a.data == b.data;
        }

        void clear() {
            data.clear();
        }

        PointHandle operator[]( size_t i ) const {
            expect(dimensions*i + dimensions <= data.size());
            PointHandle handle {
                .dimensions = dimensions,
                .vector = &data.at(dimensions*i)
            };
            return handle;
        }

        template <typename FloatIter>
        void push_back( FloatIter begin, FloatIter end ) {
            size_t cnt=0;
            for (auto it=begin; it!= end; it++)  {
                data.push_back(*it);
                cnt+=1;
            }
            expect(cnt == dimensions);
        }

        PointHandle push_back_random() {
            std::vector<float> values;
            for ( unsigned int i = 0; i < dimensions; i++ ) {
                values.push_back( sample_random_normal() );
            }

            push_back( values.begin(), values.end() );
            return operator[]( size() - 1 );
        }

        size_t size() const {
            return data.size() / dimensions;
        }
        
        size_t get_dimensions() const {
            return dimensions;
        }
    };

    class SparseSetHandle {
        size_t set_size;
        const uint32_t* tokens;

        friend class SparseSets;

    public:
        const uint32_t* begin() const {
            return tokens;
        }
        const uint32_t* end() const {
            return tokens + set_size;
        }

        size_t size() const {
            return set_size;
        }

        size_t intersection_size( SparseSetHandle other ) const {
            size_t asize = set_size;
            size_t bsize = other.set_size;

            size_t res = 0;
            size_t aidx = 0;
            size_t bidx = 0;

            while ( aidx < asize && bidx < bsize ) {
                if ( tokens[aidx] == other.tokens[bidx] ) {
                    res++;
                    aidx++;
                    bidx++;
                } else if ( tokens[aidx] < other.tokens[bidx] ) {
                    aidx++;
                } else {
                    bidx++;
                }
            }

            return res;
        }

        friend std::ostream& operator<<( std::ostream& os, const SparseSetHandle& handle ) {
            os << "{";
            for ( size_t i = 0; i < handle.set_size; i++ ) {
                os << handle.tokens[i] << ", ";
            }
            os << "}";
            return os;
        }
    };

    class SparseSets {
        size_t dimensions;
        std::vector<uint32_t> set_data;
        std::vector<size_t> starts;

    public:
        using PointHandle = SparseSetHandle;

        SparseSets( size_t dimensions ): dimensions( dimensions ) {
            starts.push_back( set_data.size() );
        }

        template <typename Archive>
        void serialize( Archive& ar ) {
            ar( dimensions, set_data, starts );
        }

        void clear() {
            set_data.clear();
            starts.clear();
            starts.push_back( set_data.size() );
        }

        PointHandle operator[]( size_t i ) const {
            PointHandle handle;
            handle.tokens = &set_data.at(starts.at(i));
            handle.set_size = starts.at(i + 1) - starts.at(i);
            return handle;
        }

        void push_back( std::vector<uint32_t>& set ) {
            std::sort( set.begin(), set.end() );
            for ( uint32_t token : set ) {
                if ( token > dimensions ) {
                    throw std::invalid_argument( "token outside of the universe" );
                }
                set_data.push_back( token );
            }
            starts.push_back( set_data.size() );
        }

        PointHandle push_back_random() {
            auto& rng = get_global_rng();
            std::bernoulli_distribution coin( 0.3 );
            std::vector<uint32_t> values;

            for ( unsigned int i = 0; i < dimensions; i++ ) {
                if ( coin( rng ) ) {
                    //
                    values.push_back( i );
                }
            }

            push_back( values );
            return operator[]( size() - 1 );
        }

        size_t size() const {
            return starts.size() - 1;
        }
    };

    template <typename Point>
    static void add_to( std::vector<float>& accumulator, const Point& point );

    template <>
    void add_to( std::vector<float>& accumulator, const EuclideanPointHandle& point ) {
        expect( accumulator.size() == point.dimensions );
        for ( size_t i = 0; i < point.dimensions; i++ ) {
            accumulator.at(i) += point.vector[i];
        }
    }

    template <>
    void add_to( std::vector<float>& accumulator, const NormedPointHandle& point ) {
        expect( accumulator.size() == point.inner.dimensions );
        std::vector<float> v(point.inner.dimensions);
        point.inner.into_vec(v);
        for ( size_t i = 0; i < point.inner.dimensions; i++ ) {
            accumulator.at(i) += v.at(i) * std::sqrt(point.sq_norm);
        }
    }

    template <typename Dataset>
    std::vector<float> mean_point( const Dataset& dataset ) {
        std::vector<float> mean( dataset.get_dimensions() );
        const size_t n = dataset.size();
        for ( size_t i = 0; i < n; i++ ) {
            add_to( mean, dataset[i] );
        }
        for ( size_t i = 0; i < dataset.get_dimensions(); i++ ) {
            mean.at(i) /= (float)n;
        }
        return mean;
    }

    //! Computes a lower bound to the diameter of the dataset
    template <typename Distance, typename Dataset>
    float approximate_diameter( Dataset& dataset ) {
        const size_t n = dataset.size();
        size_t root = 0;

        // find the farthest from arbitrary point
        float maxdist = 0.0;
        size_t maxdist_idx = 0;
        for ( size_t i = 0; i < n; i++ ) {
            float d = Distance::compute( dataset[root], dataset[i] );
            if ( d > maxdist ) {
                maxdist = d;
                maxdist_idx = i;
            }
        }

        // find the farthest from the previously found point
        maxdist = 0.0;
        for ( size_t i = 0; i < n; i++ ) {
            float d = Distance::compute( dataset[maxdist_idx], dataset[i] );
            if ( d > maxdist ) {
                maxdist = d;
            }
        }

        return maxdist;
    }

    template <typename Distance, typename Dataset>
    std::pair<std::vector<float>, std::vector<float>> distance_histogram( const Dataset& dataset,
                                                                          size_t n_bins,
                                                                          float min_distance,
                                                                          float max_distance,
                                                                          size_t sample_size ) {
        const size_t n = dataset.size();
        const size_t num_pairs = n * ( n - 1 ) / 2;
        const float sampling_factor = ( (float)num_pairs ) / sample_size;
        std::uniform_int_distribution<size_t> random_id( 0, n - 1 );
        std::vector<float> counts( n_bins + 1 );

        const double width = ( max_distance - min_distance ) / static_cast<double>( n_bins );
        std::vector<float> bounds( n_bins + 1 );
        bounds.at(0) = 0.0;
        for ( std::size_t i = 1; i < bounds.size(); ++i ) {
            bounds.at(i) = min_distance + i * width;
        }
        size_t oob = 0;
        float max_dist_found = 0.0;

#pragma omp parallel
        {
            static std::mt19937_64 rng( omp_get_thread_num() );
            std::vector<float> counts_private( n_bins + 1 );
            size_t private_oob = 0;
            float private_max_dist_found = 0.0;
#pragma omp for
            for ( size_t sample = 0; sample < sample_size; sample++ ) {
                const size_t a = random_id( rng );
                const size_t b = random_id( rng );
                const float dist = Distance::compute( dataset[a], dataset[b] );
                if ( dist < min_distance || dist > max_distance ) {
                    // ignore out of bounds, the caller does not care about them
                    private_oob++;
                    if (dist > private_max_dist_found) {
                        private_max_dist_found = dist;
                    }
                    continue;
                }
                const size_t i =
                    static_cast<size_t>( std::floor( ( dist - min_distance ) / width ) );
                if (i == counts_private.size()) {
                    LOG_INFO( "i", i, "dist", dist, "mindistance", min_distance, "width", width );
                }
                expect(i < counts_private.size());
                counts_private.at(i) += sampling_factor;
            }

#pragma omp critical
            {
                for ( size_t i = 0; i < counts_private.size(); i++ ) {
                    counts.at(i) += counts_private.at(i);
                }
                oob += private_oob;
                if (private_max_dist_found> max_dist_found) {
                    max_dist_found = private_max_dist_found ;
                }
            }
        }
        if (oob > 0) {
            LOG_WARN("msg", "out of bound pairs!", "oob", oob, "max-dist", max_dist_found);
        }

        return {counts, bounds};
    }

    struct Edge {
        float weight;
        uint32_t a;
        uint32_t b;

        friend constexpr inline bool operator<( Edge l, Edge r ) {
            return std::tie(l.weight, l.a, l.b) < std::tie(r.weight, r.a, r.b);
        }

        friend constexpr inline bool operator==( Edge l, Edge r ) {
            return std::tie(l.weight, l.a, l.b) == std::tie(r.weight, r.a, r.b);
        }
    };

} // namespace panna
