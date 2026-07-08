#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "panna/expect.hpp"

namespace panna {

    /// A HyperLogLog cardinality sketch.
    ///
    /// Fixed memory: 2^p one-byte registers (we don't pack to 6 bits; the extra
    /// factor of ~1.3 keeps the code simple and 2^14 * 1 B = 16 KiB is fine).
    /// Standard error ~ 1.04 / sqrt(2^p): p=14 gives ~0.8%.
    class HyperLogLog {
    public:
        explicit HyperLogLog( uint8_t p = 14 ):
            p_( p ), m_( size_t{ 1 } << p ), alpha_( compute_alpha( m_ ) ), registers_( m_, 0 ) {
            expect( p >= 4 && p <= 18 );
        }

        /// Add a 64-bit hash value. The caller is responsible for hashing the
        /// item into a well-mixed uint64_t (see `hash_u64` / `pair_key` below).
        void add( uint64_t hashed ) {
            const uint32_t idx = static_cast<uint32_t>( hashed >> ( 64 - p_ ) );
            // Shift the (64-p) suffix into the top of a 64-bit word and drop a
            // sentinel bit into the vacated low half so clzll is well-defined
            // even when the suffix is all zero.
            const uint64_t w = ( hashed << p_ ) | ( uint64_t{ 1 } << ( p_ - 1 ) );
            const uint8_t rank = static_cast<uint8_t>( __builtin_clzll( w ) + 1 );
            if ( rank > registers_[idx] ) {
                registers_[idx] = rank;
            }
        }

        /// Merge another sketch into this one (element-wise max of registers).
        /// Both sketches must have the same precision.
        void merge( const HyperLogLog& other ) {
            expect( p_ == other.p_ );
            for ( size_t i = 0; i < m_; i++ ) {
                if ( other.registers_[i] > registers_[i] ) {
                    registers_[i] = other.registers_[i];
                }
            }
        }

        /// Estimated number of distinct items observed.
        double estimate() const {
            double sum = 0.0;
            size_t zeros = 0;
            for ( uint8_t r : registers_ ) {
                sum += std::ldexp( 1.0, -static_cast<int>( r ) );
                if ( r == 0 ) {
                    zeros++;
                }
            }
            double e = alpha_ * static_cast<double>( m_ ) * static_cast<double>( m_ ) / sum;
            // Small-range correction: linear counting when a lot of registers
            // are still zero. For the pair counts in EMST we usually leave this
            // regime quickly, but it costs nothing to keep.
            if ( e <= 2.5 * m_ && zeros > 0 ) {
                e = static_cast<double>( m_ ) *
                    std::log( static_cast<double>( m_ ) / static_cast<double>( zeros ) );
            }
            return e;
        }

        uint8_t precision() const {
            return p_;
        }

    private:
        static double compute_alpha( size_t m ) {
            switch ( m ) {
            case 16:
                return 0.673;
            case 32:
                return 0.697;
            case 64:
                return 0.709;
            default:
                return 0.7213 / ( 1.0 + 1.079 / static_cast<double>( m ) );
            }
        }

        uint8_t p_;
        size_t m_;
        double alpha_;
        std::vector<uint8_t> registers_;
    };

    /// splitmix64 finalizer. Cheap, well-mixed, so consecutive pair-ids don't
    /// cluster into the same HLL registers.
    inline uint64_t hash_u64( uint64_t x ) {
        x += 0x9E3779B97F4A7C15ull;
        x = ( x ^ ( x >> 30 ) ) * 0xBF58476D1CE4E5B9ull;
        x = ( x ^ ( x >> 27 ) ) * 0x94D049BB133111EBull;
        x = x ^ ( x >> 31 );
        return x;
    }

    /// Pack an unordered pair (a, b) into a canonical 64-bit key.
    inline uint64_t pair_key( uint32_t a, uint32_t b ) {
        const uint32_t lo = a < b ? a : b;
        const uint32_t hi = a < b ? b : a;
        return ( static_cast<uint64_t>( hi ) << 32 ) | static_cast<uint64_t>( lo );
    }

} // namespace panna
