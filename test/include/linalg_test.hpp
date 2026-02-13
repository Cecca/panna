#pragma once

#include <catch2/catch_test_macros.hpp>

#include "panna/data.hpp"
#include "panna/linalg.hpp"

namespace panna {

    TEST_CASE( "dot_product_i16 versions equal" ) {
        size_t reps = 100;
        for ( size_t dims : { 50, 100, 128, 200, 256 } ) {

            for ( unsigned i = 0; i < reps; i++ ) {
                UnitNormPoints dataset( dims );
                dataset.push_back_random();
                dataset.push_back_random();
                UnitNormPointHandle a = dataset[0];
                UnitNormPointHandle b = dataset[1];

                int32_t simple = dot_product_chunks_simple( a, b );
#ifdef __AVX2__
                int32_t avx2 = dot_product_chunks_avx2( a, b );
                REQUIRE( simple == avx2 );
#endif
            }
        }
    }

} // namespace panna
