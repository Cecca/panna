#pragma once

#include <catch2/catch_test_macros.hpp>

#include "panna/data.hpp"
#include "panna/linalg.hpp"
#include "panna/rand.hpp"

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

                int16_t simple = dot_product_chunks16_simple( a, b );
#ifdef __AVX2__
                int16_t avx2 = dot_product_chunks16_avx2( a, b );
                REQUIRE( simple == avx2 );
#endif
            }
        }
    }

    TEST_CASE( "euclidean distance" ) {
        size_t reps = 100;
        for ( size_t dims : { 50, 100, 128, 200, 256, 784 } ) {
            for ( unsigned i = 0; i < reps; i++ ) {
                std::vector<float> a = sample_random_normal_vector(dims);
                std::vector<float> b = sample_random_normal_vector(dims);

                float simple = euclidean_naive( a.data(), b.data(), dims );
#ifdef __AVX2__
                float avx2 = euclidean_avx2( a.data(), b.data(), dims );
                REQUIRE( std::abs( simple - avx2 ) <= 1e-4 );
#endif
            }
        }
    }

    TEST_CASE( "pseudorandom rotations" ) {

        int d = 100;     // Input vector dimension
        int L = 1 << 14; // Output dimension (must be power of 2): 2^14 = 16384

        std::cout << "Input dimension:  " << d << std::endl;
        std::cout << "Output dimension: " << L << std::endl << std::endl;

        std::cout << "Initializing rotator..." << std::endl;
        RandomDotProducts rdps( L );
        std::cout << "Done.\n\n";

        // Create a random input vector (d-dimensional)
        std::vector<float> x( d );
        for ( int i = 0; i < d; ++i ) {
            x[i] = std::sin( i * 0.1 ); // Simple deterministic vector
        }

        float input_norm = std::sqrt(dot_product( x, x ));
        std::cout << "Input vector ||x|| = " << input_norm << std::endl;

        std::cout << "\nApplying random rotation: y = H*D3*H*D2*H*D1*[x; 0, ..., 0]" << std::endl;
        auto start = std::chrono::high_resolution_clock::now();
        std::vector<float> y(L, 0);
        std::copy(x.begin(), x.end(), y.begin());
        rdps.compute( y );
        auto end = std::chrono::high_resolution_clock::now();
        auto duration_us =
            std::chrono::duration_cast<std::chrono::microseconds>( end - start ).count();

        float output_norm = std::sqrt(dot_product( y, y ));
        std::cout << "Output vector ||y|| = " << output_norm << std::endl;
        std::cout << "Time: " << duration_us / 1000.0 << " ms" << std::endl << std::endl;

        // Verify norm preservation (approximately)
        std::cout << "Norm preservation check:" << std::endl;
        std::cout << "  Input norm:  " << input_norm << std::endl;
        std::cout << "  Output norm: " << output_norm << std::endl;
        std::cout << "  Ratio (output/input): " << output_norm / input_norm << std::endl;

        // Second vector for demonstrating angle preservation
        std::vector<float> x2( d );
        for ( int i = 0; i < d; ++i ) {
            x2[i] = std::cos( i * 0.1 );
        }

        std::vector<float> y2(L, 0.0);
        std::copy(x2.begin(), x2.end(), y2.begin());
        rdps.compute( y2 );

        float dot_before = dot_product( x, x2 );
        float dot_after = dot_product( y, y2 );

        float norm_x2 = std::sqrt(dot_product( x2, x2 ));
        float norm_y2 = std::sqrt(dot_product( y2, y2 ));

        float angle_before = std::acos( dot_before / ( input_norm * norm_x2 ) );
        float angle_after = std::acos( dot_after / ( output_norm * norm_y2 ) );

        std::cout << "Angle preservation (for two vectors):" << std::endl;
        std::cout << "  Angle before: " << angle_before * 180.0 / M_PI << " degrees" << std::endl;
        std::cout << "  Angle after:  " << angle_after * 180.0 / M_PI << " degrees" << std::endl;

        // Show first few components
        std::cout << "First 10 components of rotated vector:" << std::endl;
        for ( int i = 0; i < 10; ++i ) {
            std::cout << "  y[" << i << "] = " << std::setw( 10 ) << y[i] << std::endl;
        }
        REQUIRE(std::abs(input_norm - output_norm) <= 1e-5);
    }

    TEST_CASE( "dot product distribution" ) {
        using Vec = std::vector<float>;
        size_t d = 128;
        size_t L = 1 << 10;
        Vec x = sample_random_normal_vector(d);
        // normalize(x);

        std::vector<float> empirical_distribution;
        for(size_t i=0; i<L; i++) {
            Vec dir = sample_random_normal_vector(d);
            // normalize(dir);
            float dp = dot_product(x, dir);
            empirical_distribution.push_back(dp);
        }

        RandomDotProducts rdps(L);
        Vec hadamard = rdps.allocate_scratch();
        std::copy(x.begin(), x.end(), hadamard.begin());
        rdps.compute(hadamard);

        std::ranges::sort(empirical_distribution);
        std::ranges::sort(hadamard);
        for (size_t i=0; i<L; i++) {
            std::cout << empirical_distribution[i] << " " << hadamard[i] << std::endl;
        }
    }

    //! Build the Gram matrix `A A^T` of a random `n x d` matrix, which is
    //! symmetric positive definite as long as `n <= d`.
    static std::vector<float> random_gram_matrix( size_t n, size_t d ) {
        std::vector<std::vector<float>> rows;
        for ( size_t i = 0; i < n; i++ ) {
            rows.push_back( sample_random_normal_vector( d ) );
        }
        std::vector<float> gram( n * n );
        for ( size_t i = 0; i < n; i++ ) {
            for ( size_t j = 0; j < n; j++ ) {
                gram.at( i * n + j ) = dot_product( rows.at( i ), rows.at( j ) ) / d;
            }
        }
        return gram;
    }

    TEST_CASE( "cholesky decomposition" ) {
        seed_global_rng( 1234 );
        const size_t d = 512;
        for ( size_t n : { 1, 2, 5, 8, 16, 32 } ) {
            std::vector<float> gram = random_gram_matrix( n, d );
            CholeskyFactor factor( gram, n );
            REQUIRE( factor.get_dimensions() == n );

            for ( size_t i = 0; i < n; i++ ) {
                REQUIRE( factor( i, i ) > 0.0f );
                for ( size_t j = i + 1; j < n; j++ ) {
                    REQUIRE( factor( i, j ) == 0.0f );
                }
            }

            // Check that L L^T reconstructs the original matrix.
            for ( size_t i = 0; i < n; i++ ) {
                for ( size_t j = 0; j < n; j++ ) {
                    float reconstructed = 0.0f;
                    for ( size_t k = 0; k < n; k++ ) {
                        reconstructed += factor( i, k ) * factor( j, k );
                    }
                    REQUIRE( std::abs( reconstructed - gram.at( i * n + j ) ) <= 1e-4 );
                }
            }
        }
    }

    TEST_CASE( "cholesky rejects non positive definite matrices" ) {
        // A matrix with two identical rows is singular, hence not positive
        // definite.
        std::vector<float> singular = { 1.0f, 1.0f, 1.0f, 1.0f };
        REQUIRE_THROWS_AS( CholeskyFactor( singular, 2 ), std::invalid_argument );
    }

    TEST_CASE( "cholesky multiply" ) {
        seed_global_rng( 1234 );
        const size_t d = 512;
        for ( size_t n : { 1, 3, 16 } ) {
            CholeskyFactor factor( random_gram_matrix( n, d ), n );
            std::vector<float> in = sample_random_normal_vector( n );
            std::vector<float> out;
            factor.multiply( in, out );
            REQUIRE( out.size() == n );

            for ( size_t i = 0; i < n; i++ ) {
                float expected = 0.0f;
                for ( size_t j = 0; j < n; j++ ) {
                    expected += factor( i, j ) * in.at( j );
                }
                REQUIRE( std::abs( out.at( i ) - expected ) <= 1e-5 );
            }
        }
    }

    //! The covariance of `L z`, with `z` standard normal, must be the matrix
    //! that was decomposed. This catches a transposed factor, which the
    //! reconstruction check above cannot see.
    TEST_CASE( "cholesky sampling covariance" ) {
        seed_global_rng( 1234 );
        const size_t n = 4;
        const size_t samples = 200000;
        std::vector<float> gram = random_gram_matrix( n, 512 );
        CholeskyFactor factor( gram, n );

        std::vector<float> z( n );
        std::vector<float> y;
        std::vector<double> covariance( n * n, 0.0 );
        for ( size_t s = 0; s < samples; s++ ) {
            for ( size_t i = 0; i < n; i++ ) {
                z.at( i ) = sample_random_normal();
            }
            factor.multiply( z, y );
            for ( size_t i = 0; i < n; i++ ) {
                for ( size_t j = 0; j < n; j++ ) {
                    covariance.at( i * n + j ) += y.at( i ) * y.at( j );
                }
            }
        }

        for ( size_t i = 0; i < n; i++ ) {
            for ( size_t j = 0; j < n; j++ ) {
                double empirical = covariance.at( i * n + j ) / samples;
                REQUIRE( std::abs( empirical - gram.at( i * n + j ) ) <= 0.02 );
            }
        }
    }

} // namespace panna
