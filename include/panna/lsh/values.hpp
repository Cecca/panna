#pragma once

#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <ostream>
#include <type_traits>
#include <functional>

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
        friend struct std::hash<BitwiseLshValue<K>>;

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

        //! Copy the bit at concatenation index `pos_from` of `other` into
        //! concatenation index `pos_to` of `this`. Concatenation index `i`
        //! is stored at bit `K - 1 - i`, consistently with `prefix_eq`.
        void overwrite( const BitwiseLshValue<K>& other, uint8_t pos_from, uint8_t pos_to ) {
            assert( pos_from < K );
            assert( pos_to < K );
            const uint32_t bit = ( other.bits >> ( K - 1 - pos_from ) ) & 1;
            const uint32_t mask = uint32_t( 1 ) << ( K - 1 - pos_to );
            bits = ( bits & ~mask ) | ( bit << ( K - 1 - pos_to ) );
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

        friend struct std::hash<SymbolLshValue<Symbol, K>>;

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

        void overwrite(const SymbolLshValue<Symbol, K> &other, uint8_t pos_from, uint8_t pos_to) {
            assert( pos_from <= K );
            assert( pos_to <= K );
            hashes[pos_to] = other.hashes[pos_from];
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
                os << "[";
                for ( size_t j = 0; j < SIZE; j++ ) {
                    os << std::hex << +hash.hashes[i][j];
                    if ( j < SIZE - 1 ) {
                        os << ",";
                    }
                }
                os << "]";
                if ( i < K - 1 ) {
                    os << "_";
                }
            }
            return os;
        }

    public:
        using DoubleWidth = ArrayLshValue<T, SIZE, 2 * K>;

        friend struct std::hash<ArrayLshValue<T, SIZE, K>>;

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


template<typename Symbol, uint8_t K>
struct std::hash<panna::SymbolLshValue<Symbol, K>>
{
    std::size_t operator()(const panna::SymbolLshValue<Symbol, K>& s) const noexcept
    {
        std::size_t hval = 0;
        for ( uint8_t i = 0; i < K; i++ ) {
            hval ^= std::hash<Symbol>{}( s.hashes[i] ) + 0x9e3779b97f4a7c15ULL
                  + ( hval << 6 ) + ( hval >> 2 );
        }
        return hval;
    }
};

template<typename T, size_t SIZE, uint8_t K>
struct std::hash<panna::ArrayLshValue<T, SIZE, K>>
{
    std::size_t operator()(const panna::ArrayLshValue<T, SIZE, K>& s) const noexcept
    {
        std::size_t hval = 0;
        for ( uint8_t i = 0; i < K; i++ ) {
            for ( size_t j = 0; j < SIZE; j++ ) {
                hval ^= std::hash<T>{}( s.hashes[i][j] ) + 0x9e3779b97f4a7c15ULL
                      + ( hval << 6 ) + ( hval >> 2 );
            }
        }
        return hval;
    }
};

template<uint8_t K>
struct std::hash<panna::BitwiseLshValue<K>>
{
    std::size_t operator()(const panna::BitwiseLshValue<K>& s) const noexcept
    {
        std::size_t hval = std::hash<uint32_t>{}(s.bits);
        return hval;
    }
};

