#pragma once
#include <catch2/catch_test_macros.hpp>

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
} // namespace panna
