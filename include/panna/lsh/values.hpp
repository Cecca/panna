#pragma once

#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <ostream>
#include <type_traits>

namespace panna {
    //! Places zeros between the bits of the given value.
    //! Only the 16 less significant bits will appear in the output.
    //! The least significant bit will remain where it is.
    constexpr static uint32_t intersperse_zero( uint32_t val ) {
        uint32_t mask = 1;
        uint32_t shift = 0;

        uint32_t res = 0;
        for ( unsigned i = 0; i < sizeof( uint32_t ) * 8 / 2; i++ ) {
            res |= ( val & mask ) << shift;
            mask <<= 1;
            shift++;
        }
        return res;
    }
    // An example of a call to the function above
    static_assert( intersperse_zero( 0xFF ) == 0x5555, "" );

    // clang-format off
    static constexpr uint32_t BITWISE_HASH_MASKS[] = {
        0b00000000000000000000000000000000,
        0b10000000000000000000000000000000,
        0b11000000000000000000000000000000,
        0b11100000000000000000000000000000,
        0b11110000000000000000000000000000,
        0b11111000000000000000000000000000,
        0b11111100000000000000000000000000,
        0b11111110000000000000000000000000,
        0b11111111000000000000000000000000,
        0b11111111100000000000000000000000,
        0b11111111110000000000000000000000,
        0b11111111111000000000000000000000,
        0b11111111111100000000000000000000,
        0b11111111111110000000000000000000,
        0b11111111111111000000000000000000,
        0b11111111111111100000000000000000,
        0b11111111111111110000000000000000,
        0b11111111111111111000000000000000,
        0b11111111111111111100000000000000,
        0b11111111111111111110000000000000,
        0b11111111111111111111000000000000,
        0b11111111111111111111100000000000,
        0b11111111111111111111110000000000,
        0b11111111111111111111111000000000,
        0b11111111111111111111111100000000,
        0b11111111111111111111111110000000,
        0b11111111111111111111111111000000,
        0b11111111111111111111111111100000,
        0b11111111111111111111111111110000,
        0b11111111111111111111111111111000,
        0b11111111111111111111111111111100,
        0b11111111111111111111111111111110,
        0b11111111111111111111111111111111,
    };
    // clang-format on

    template <uint8_t K>
    struct BitwiseLshValue {
    private:
        static_assert( K <= 32, "you can have at most 32 hash functions" );
        //! The bits of this hash value
        uint32_t bits;

        // To allow the implementation of `interleave` to work
        friend struct BitwiseLshValue<2 * K>;

    public:
        using DoubleWidth = BitwiseLshValue<2 * K>;

        //! How many concatenated hash values are stored in this value?
        constexpr static uint8_t get_concatenations() {
            return K;
        };

        template <typename Archive>
        void serialize( Archive& ar ) {
            ar( bits );
        }

        // we use a factory function rather than a constructor to keep this
        // struct Plain Old Data.
        constexpr static BitwiseLshValue<K> make( uint32_t bits ) {
            BitwiseLshValue<K> val;
            val.bits = bits;
            return val;
        }

        //! Check if the given hash value and `this` have the same prefix of
        //! bits, with the understanding that the higher order 32-BITS bits are
        //! not used, hence the prefix starts from that position.
        inline bool prefix_eq( BitwiseLshValue<K> other, uint8_t prefix ) const {
            auto idx = 32 - ( K - prefix );
            assert( idx <= 32 );
            uint32_t mask = BITWISE_HASH_MASKS[idx];
            return ( this->bits & mask ) == ( other.bits & mask );
        }

        //! Check if the prefix of `this` is `<` than the prefix of `other`.
        //! with the understanding that the higher order 32-BITS bits are not
        //! used, hence the prefix starts from that position.
        inline bool prefix_less( BitwiseLshValue<K> other, uint8_t prefix ) const {
            auto idx = 32 - ( K - prefix );
            assert( idx <= 32 );
            uint32_t mask = BITWISE_HASH_MASKS[idx];
            return ( this->bits & mask ) < ( other.bits & mask );
        }

        //! Hash values are ordered lexicographically, from the most to the
        //! least significant bits.
        constexpr inline bool operator<( BitwiseLshValue<K> other ) const {
            return this->bits < other.bits;
        }

        //! This equality operator considers all the bits.
        constexpr inline bool operator==( BitwiseLshValue<K> other ) const {
            return this->bits == other.bits;
        }

        static constexpr inline BitwiseLshValue<K> interleave( BitwiseLshValue<K / 2> a,
                                                               BitwiseLshValue<K / 2> b ) {
            uint32_t abits = intersperse_zero( a.bits );
            uint32_t bbits = intersperse_zero( b.bits );
            uint32_t res = ( abits << 1 ) | bbits;
            return BitwiseLshValue<K>::make( res );
        }

        /// Number of bytes in the radix sort key.
        static constexpr size_t radix_key_bytes() { return 4; }

        /// Extract one byte for LSD radix sort.
        /// Pass 0 = least significant byte in the sort order.
        /// BitwiseLshValue wraps an unsigned uint32_t;
        /// on little-endian x86, byte 0 is already the LSB.
        uint8_t radix_key_byte( size_t pass ) const {
            return reinterpret_cast<const uint8_t*>( &bits )[pass];
        }
    };

    // simple static tests on the hash data type, checked at compile time
    static_assert( sizeof( BitwiseLshValue<24> ) == 4, "BitwiseLshValue must be 4 bytes" );
    static_assert( std::is_standard_layout<BitwiseLshValue<24>>(), "BitwiseLshValue must be standard layout" );
    static_assert( std::is_trivially_copyable<BitwiseLshValue<24>>(),
                   "BitwiseLshValue must be trivially copiable" );
    static_assert( std::is_trivial<BitwiseLshValue<24>>(), "BitwiseLshValue must be trivial" );

    template <typename Symbol, uint8_t K>
    struct SymbolLshValue {
    private:
        //! The hash values
        std::array<Symbol, K> hashes;

        // To allow the implementation of `interleave` to work
        friend struct SymbolLshValue<Symbol, 2 * K>;
        friend std::ostream& operator<<( std::ostream& os, const SymbolLshValue<Symbol, K>& hash ) {
            os << "#";
            for ( size_t i = 0; i < K; i++ ) {
                os << std::hex << +hash.hashes[i];
                // os << std::hex << static_cast<int64_t>( hash.hashes[i] );
                if ( i < K - 1 ) {
                    os << "_";
                }
            }
            return os;
        }

    public:
        using DoubleWidth = SymbolLshValue<Symbol, 2 * K>;

        template <typename Archive>
        void serialize( Archive& ar ) {
            ar( hashes );
        }

        //! How many concatenated hash values are stored in this value?
        constexpr static uint8_t get_concatenations() {
            return K;
        };

        // we use a factory function rather than a constructor to keep this
        // struct Plain Old Data.
        constexpr static SymbolLshValue<Symbol, K> make( std::array<Symbol, K> bytes ) {
            SymbolLshValue<Symbol, K> hash;
            hash.hashes = bytes;
            return hash;
        }

        void set( size_t idx, Symbol value ) {
            hashes[idx] = value;
        }

        inline bool prefix_eq( SymbolLshValue<Symbol, K> other, uint8_t prefix ) const {
            assert( prefix <= K );
            for ( uint8_t i = 0; i < prefix; i++ ) {
                if ( hashes[i] != other.hashes[i] ) {
                    return false;
                }
            }
            return true;
        }

        inline constexpr bool prefix_less( const SymbolLshValue<Symbol, K>& other,
                                           uint8_t prefix ) const {
            assert( prefix <= K );
            for ( uint8_t i = 0; i < prefix; i++ ) {
                if ( hashes[i] < other.hashes[i] ) {
                    return true;
                } else if ( hashes[i] > other.hashes[i] ) {
                    return false;
                }
            }
            return false;
        }

        constexpr inline bool operator<( SymbolLshValue<Symbol, K> other ) const {
            return this->prefix_less( other, K );
        }

        constexpr inline bool operator==( SymbolLshValue<Symbol, K> other ) const {
            return this->prefix_eq( other, K );
        }

        /// Number of bytes in the radix sort key.
        static constexpr size_t radix_key_bytes() { return K * sizeof(Symbol); }

        /// Extract one byte for LSD radix sort.
        /// Pass 0 = least significant byte in the sort order.
        /// Elements are in lexicographic order (element 0 = most significant).
        /// On little-endian x86 each element has LSB at offset 0.
        /// For signed Symbol types the MSB of each element is sign-flipped
        /// so that unsigned byte comparison matches signed comparison.
        uint8_t radix_key_byte( size_t pass ) const {
            constexpr size_t ELEM_SIZE = sizeof(Symbol);
            // LSD pass ordering: element K-1 bytes first, then K-2, ..., then 0
            size_t element = K - 1 - pass / ELEM_SIZE;
            size_t byte_within = pass % ELEM_SIZE;
            const uint8_t* raw = reinterpret_cast<const uint8_t*>( &hashes[element] );
            uint8_t b = raw[byte_within];
            // Flip sign bit on MSB of each element for signed types
            if constexpr ( std::is_signed_v<Symbol> ) {
                if ( byte_within == ELEM_SIZE - 1 ) {
                    b ^= 0x80;
                }
            }
            return b;
        }

        static constexpr inline SymbolLshValue<Symbol, K>
        interleave( SymbolLshValue<Symbol, K / 2> a, SymbolLshValue<Symbol, K / 2> b ) {
            static_assert( K % 2 == 0, "K should be even" );
            SymbolLshValue<Symbol, K> out;
            for ( size_t i = 0; i < K / 2; i++ ) {
                out.hashes[2 * i] = a.hashes[i];
                out.hashes[2 * i + 1] = b.hashes[i];
            }
            return out;
        }
    };

    template <uint8_t K>
    using BytewiseLshValue = SymbolLshValue<int8_t, K>;

    template <uint8_t K>
    using ShortLshValue = SymbolLshValue<int16_t, K>;

    template <uint8_t K>
    using IntLshValue = SymbolLshValue<int32_t, K>;

    template <uint8_t K>
    using LongLshValue = SymbolLshValue<int64_t, K>;

    static_assert( sizeof( BytewiseLshValue<8> ) == 8,
                   "BytewiseLshValue must use exactly K bytes" );
    static_assert( std::is_standard_layout<BytewiseLshValue<8>>(), "BytewiseLshValue must be Plain Old Data" );
    static_assert( std::is_trivially_copyable<BytewiseLshValue<8>>(),
                   "BytewiseLshValue must be trivially copiable" );
    static_assert( std::is_trivial<BytewiseLshValue<8>>(), "BytewiseLshValue must be trivial" );

    static_assert( sizeof( ShortLshValue<4> ) == 8, "ShortLshValue must use exactly K bytes" );
    static_assert( std::is_standard_layout<ShortLshValue<4>>(), "ShortLshValue must be Plain Old Data" );
    static_assert( std::is_trivially_copyable<ShortLshValue<4>>(),
                   "ShortLshValue must be trivially copiable" );
    static_assert( std::is_trivial<ShortLshValue<4>>(), "ShortLshValue must be trivial" );

    template <typename T, size_t SIZE, uint8_t K>
    struct ArrayLshValue {
    private:
        //! The hash values
        std::array<std::array<T, SIZE>, K> hashes;

        // To allow the implementation of `interleave` to work
        friend struct ArrayLshValue<T, SIZE, 2 * K>;

        friend std::ostream& operator<<( std::ostream& os, const ArrayLshValue<T, SIZE, K>& hash ) {
            os << "#";
            for ( size_t i = 0; i < K; i++ ) {
                os << std::hex << +hash.hashes[i];
                // os << std::hex << static_cast<int64_t>( hash.hashes[i] );
                if ( i < K - 1 ) {
                    os << "_";
                }
            }
            return os;
        }

    public:
        using DoubleWidth = ArrayLshValue<T, SIZE, 2 * K>;

        template <typename Archive>
        void serialize( Archive& ar ) {
            ar( hashes );
        }

        //! How many concatenated hash values are stored in this value?
        constexpr static uint8_t get_concatenations() {
            return K;
        };

        // we use a factory function rather than a constructor to keep this
        // struct Plain Old Data.
        constexpr static ArrayLshValue<T, SIZE, K> make( std::array<std::array<T, SIZE>, K> bytes ) {
            ArrayLshValue<T, SIZE, K> hash;
            hash.hashes = bytes;
            return hash;
        }

        void set( size_t idx, uint8_t value ) {
            hashes[idx] = value;
        }

        inline bool prefix_eq( ArrayLshValue<T, SIZE, K> other, uint8_t prefix ) const {
            assert( prefix <= K );
            for ( uint8_t i = 0; i < prefix; i++ ) {
                if ( hashes[i] != other.hashes[i] ) {
                    return false;
                }
            }
            return true;
        }

        inline constexpr bool prefix_less( const ArrayLshValue<T, SIZE, K>& other,
                                           uint8_t prefix ) const {
            assert( prefix <= K );
            for ( uint8_t i = 0; i < prefix; i++ ) {
                if ( hashes[i] < other.hashes[i] ) {
                    return true;
                } else if ( hashes[i] > other.hashes[i] ) {
                    return false;
                }
            }
            return false;
        }

        constexpr inline bool operator<( ArrayLshValue<T, SIZE, K> other ) const {
            return this->prefix_less( other, K );
        }

        constexpr inline bool operator==( ArrayLshValue<T, SIZE, K> other ) const {
            return this->prefix_eq( other, K );
        }

        static constexpr inline ArrayLshValue<T, SIZE, K>
        interleave( ArrayLshValue<T, SIZE, K / 2> a, ArrayLshValue<T, SIZE, K / 2> b ) {
            static_assert( K % 2 == 0, "K should be even" );
            ArrayLshValue<T, SIZE, K> out;
            for ( size_t i = 0; i < K / 2; i++ ) {
                out.hashes[2 * i] = a.hashes[i];
                out.hashes[2 * i + 1] = b.hashes[i];
            }
            return out;
        }
    };
} // namespace panna
