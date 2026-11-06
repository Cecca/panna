#pragma once
#include <catch2/catch_test_macros.hpp>

#include <random>
#include <unordered_set>

#include "panna/lsh/values.hpp"

namespace panna {
    TEST_CASE( "prefix_less" ) {
        ShortLshValue<4> a = ShortLshValue<4>::make( { 0x7d, 0x4, 0x56, 0x74 } );
        ShortLshValue<4> b = ShortLshValue<4>::make( { 0x7d, 0x8, 0x78, 0x46 } );
        ShortLshValue<4> c = ShortLshValue<4>::make( { 0x7d, 0x16, 0x83, 0xe8 } );

        REQUIRE( a.prefix_less( b, 4 ) );
        REQUIRE( b.prefix_less( c, 4 ) );

        ShortLshValue<4> x = ShortLshValue<4>::make( { 0x61, 0x0, 0x0, 0x0 } );
        ShortLshValue<4> y = ShortLshValue<4>::make( { 0x1, 0x2f, 0xbd, 0xbf } );
        REQUIRE( !x.prefix_less(y, 4) );

    }

    TEST_CASE( "Bit hashes interleaving" ) {
        BitwiseLshValue<16> a = BitwiseLshValue<16>::make( 0xaaaa );
        BitwiseLshValue<16> b = BitwiseLshValue<16>::make( 0xaaaa );
        BitwiseLshValue<32> expected = BitwiseLshValue<32>::make( 0xcccccccc );
        REQUIRE( BitwiseLshValue<32>::interleave( a, b ) == expected );
    }

    TEST_CASE( "Byte hashes interleaving" ) {
        std::array<int8_t, 4> bytes_a = { 0, 2, 4, 6 };
        std::array<int8_t, 4> bytes_b = { 1, 3, 5, 7 };
        std::array<int8_t, 8> bytes_expected = { 0, 1, 2, 3, 4, 5, 6, 7 };
        BytewiseLshValue<4> a = BytewiseLshValue<4>::make( bytes_a );
        BytewiseLshValue<4> b = BytewiseLshValue<4>::make( bytes_b );
        BytewiseLshValue<8> expected = BytewiseLshValue<8>::make( bytes_expected );
        REQUIRE( BytewiseLshValue<8>::interleave( a, b ) == expected );
    }

    TEST_CASE( "hash values can be hashed" ) {
        std::array<int8_t, 4> bytes_a = { 0, 2, 4, 6 };
        std::array<int8_t, 4> bytes_b = { 1, 3, 5, 7 };
        BytewiseLshValue<4> a = BytewiseLshValue<4>::make( bytes_a );
        BytewiseLshValue<4> b = BytewiseLshValue<4>::make( bytes_b );
        REQUIRE( std::hash<BytewiseLshValue<4>>{}(a) != std::hash<BytewiseLshValue<4>>{}(b)  );
        REQUIRE( std::hash<BytewiseLshValue<4>>{}(a) ==
                 std::hash<BytewiseLshValue<4>>{}( BytewiseLshValue<4>::make( bytes_a ) ) );

        std::unordered_set<BytewiseLshValue<4>> set;
        set.insert( a );
        set.insert( b );
        set.insert( BytewiseLshValue<4>::make( bytes_a ) );
        REQUIRE( set.size() == 2 );
        REQUIRE( set.count( a ) == 1 );
    }

    TEST_CASE( "array hash values can be hashed" ) {
        using Value = ArrayLshValue<int32_t, 3, 2>;
        std::array<std::array<int32_t, 3>, 2> arrays_a = { { { 0, 1, 2 }, { 3, 4, 5 } } };
        std::array<std::array<int32_t, 3>, 2> arrays_b = { { { 0, 1, 2 }, { 3, 4, 6 } } };
        Value a = Value::make( arrays_a );
        Value b = Value::make( arrays_b );
        REQUIRE( std::hash<Value>{}( a ) != std::hash<Value>{}( b ) );
        REQUIRE( std::hash<Value>{}( a ) == std::hash<Value>{}( Value::make( arrays_a ) ) );
        REQUIRE( a == Value::make( arrays_a ) );
        REQUIRE( !( a == b ) );

        std::unordered_set<Value> set;
        set.insert( a );
        set.insert( b );
        set.insert( Value::make( arrays_a ) );
        REQUIRE( set.size() == 2 );
        REQUIRE( set.count( a ) == 1 );
    }

    TEST_CASE( "bitwise hash values can be hashed" ) {
        BitwiseLshValue<16> a = BitwiseLshValue<16>::make( 0xaaaa );
        BitwiseLshValue<16> b = BitwiseLshValue<16>::make( 0x5555 );
        REQUIRE( std::hash<BitwiseLshValue<16>>{}( a ) != std::hash<BitwiseLshValue<16>>{}( b ) );
        REQUIRE( std::hash<BitwiseLshValue<16>>{}( a ) ==
                 std::hash<BitwiseLshValue<16>>{}( BitwiseLshValue<16>::make( 0xaaaa ) ) );

        std::unordered_set<BitwiseLshValue<16>> set;
        set.insert( a );
        set.insert( b );
        set.insert( BitwiseLshValue<16>::make( 0xaaaa ) );
        REQUIRE( set.size() == 2 );
        REQUIRE( set.count( a ) == 1 );
    }

    //! Check that `get_bits` agrees with `prefix_eq`: concatenation `i` is bit `K - 1 - i`.
    template <uint8_t K>
    static void check_bitwise_accessors( std::mt19937_64& rng ) {
        const uint32_t low_mask = K == 32 ? 0xFFFFFFFFu : ( ( uint32_t( 1 ) << K ) - 1 );
        auto bit = []( BitwiseLshValue<K> v, uint8_t i ) { return ( v.get_bits() >> ( K - 1 - i ) ) & 1; };
        for ( size_t trial = 0; trial < 1000; trial++ ) {
            const uint32_t abits = static_cast<uint32_t>( rng() ) & low_mask;
            // flip a random subset of the bits after a random position, so
            // that the two values share prefixes of any length
            const uint8_t from = std::uniform_int_distribution<int>( 0, K )( rng );
            const uint32_t suffix_mask = from == 0 ? low_mask : ( ( uint32_t( 1 ) << ( K - from ) ) - 1 );
            const uint32_t bbits = abits ^ ( static_cast<uint32_t>( rng() ) & suffix_mask );
            auto a = BitwiseLshValue<K>::make( abits );
            auto b = BitwiseLshValue<K>::make( bbits );
            REQUIRE( a.get_bits() == abits );
            REQUIRE( b.get_bits() == bbits );
            for ( uint8_t p = 0; p <= K; p++ ) {
                bool eq = true;
                for ( uint8_t i = 0; i < p; i++ ) {
                    eq = eq && bit( a, i ) == bit( b, i );
                }
                REQUIRE( a.prefix_eq( b, p ) == eq );
            }
            // `overwrite` writes concatenation `pos_to`, as read by `get_bits`
            const uint8_t pos_from = std::uniform_int_distribution<int>( 0, K - 1 )( rng );
            const uint8_t pos_to = std::uniform_int_distribution<int>( 0, K - 1 )( rng );
            auto c = a;
            c.overwrite( b, pos_from, pos_to );
            for ( uint8_t i = 0; i < K; i++ ) {
                REQUIRE( bit( c, i ) == ( i == pos_to ? bit( b, pos_from ) : bit( a, i ) ) );
            }
        }
    }

    //! Check that `get` and `data` agree with `make` and with `prefix_eq`.
    template <typename Symbol, uint8_t K>
    static void check_symbol_accessors( std::mt19937_64& rng ) {
        using Value = SymbolLshValue<Symbol, K>;
        // a tiny alphabet, so that random suffixes collide often
        std::uniform_int_distribution<int> symbol( -1, 1 );
        for ( size_t trial = 0; trial < 1000; trial++ ) {
            std::array<Symbol, K> asym, bsym;
            for ( size_t i = 0; i < K; i++ ) {
                asym[i] = static_cast<Symbol>( symbol( rng ) );
            }
            const uint8_t from = std::uniform_int_distribution<int>( 0, K )( rng );
            for ( size_t i = 0; i < K; i++ ) {
                bsym[i] = i < from ? asym[i] : static_cast<Symbol>( symbol( rng ) );
            }
            Value a = Value::make( asym );
            Value b = Value::make( bsym );
            for ( size_t i = 0; i < K; i++ ) {
                REQUIRE( a.get( i ) == asym[i] );
                REQUIRE( b.get( i ) == bsym[i] );
                REQUIRE( a.data()[i] == asym[i] );
                REQUIRE( b.data()[i] == bsym[i] );
            }
            for ( uint8_t p = 0; p <= K; p++ ) {
                bool eq = true;
                for ( uint8_t i = 0; i < p; i++ ) {
                    eq = eq && a.get( i ) == b.get( i );
                }
                REQUIRE( a.prefix_eq( b, p ) == eq );
            }
        }
    }

    TEST_CASE( "BitwiseLshValue::get_bits agrees with prefix_eq", "[lsh][values]" ) {
        std::mt19937_64 rng( 1234 );
        check_bitwise_accessors<1>( rng );
        check_bitwise_accessors<2>( rng );
        check_bitwise_accessors<5>( rng );
        check_bitwise_accessors<8>( rng );
        check_bitwise_accessors<16>( rng );
        check_bitwise_accessors<31>( rng );
        check_bitwise_accessors<32>( rng );
    }

    TEST_CASE( "SymbolLshValue::get agrees with prefix_eq", "[lsh][values]" ) {
        std::mt19937_64 rng( 1234 );
        check_symbol_accessors<int8_t, 1>( rng );
        check_symbol_accessors<int8_t, 8>( rng );
        check_symbol_accessors<int16_t, 1>( rng );
        check_symbol_accessors<int16_t, 4>( rng );
        check_symbol_accessors<int32_t, 2>( rng );
        check_symbol_accessors<int32_t, 16>( rng );
        check_symbol_accessors<int64_t, 1>( rng );
        check_symbol_accessors<int64_t, 3>( rng );
        check_symbol_accessors<int64_t, 16>( rng );
    }
} // namespace panna
