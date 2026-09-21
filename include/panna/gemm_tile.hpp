#pragma once

//! A GEMM-based *rejection filter* for the tile kernel of `PairForestIndex`.
//!
//! A tile asks for up to `128 x 128` distances between two runs of sorted
//! positions. Computed one pair at a time, every distance streams both points
//! from memory; computed as one matrix product, the row and column blocks stay
//! in cache and the arithmetic runs at GEMM speed. The catch is accuracy: the
//! product gives `|a|^2 + |b|^2 - 2 a.b`, which cancels catastrophically for
//! exactly the *close* pairs the EMST search is after.
//!
//! So the GEMM value never becomes an edge weight. It is only used to prove
//! that a pair is too far to pass the threshold, with a rounding-error bound
//! wide enough to cover both the estimate and the scalar `Distance::compute`
//! that the pair would otherwise have gone through. Every pair that survives
//! is recomputed with `Distance::compute`, so the edges emitted, their order
//! and the distance count are bit-identical to those of the scalar kernel.
//!
//! The kernel is opt-in per `(Dataset, Distance)` through `GemmTileKernel`.
//! Only `EuclideanPoints` (already contiguous, row-major float) with the two
//! Euclidean distances is enabled. `UnitNormPoints` / `NormedPoints` store
//! int16 chunks and would need a conversion or an int16 GEMM first, and
//! `JaccardDistance` over `SparseSets` has no inner-product form at all.

/// The tile kernels run inside the caller's OpenMP threads: a GEMM that spawned
/// threads of its own would oversubscribe the machine. CMake also defines this
/// for every target; the guard covers users that include the header directly.
/// It has to come before the first Eigen include of the translation unit.
#if defined( EIGEN_CORE_H ) && !defined( EIGEN_DONT_PARALLELIZE )
    #warning "panna/gemm_tile.hpp: Eigen was included before it without EIGEN_DONT_PARALLELIZE; \
the tile GEMM may use OpenMP threads of its own. Define it before including Eigen."
#endif
#ifndef EIGEN_DONT_PARALLELIZE
    #define EIGEN_DONT_PARALLELIZE
#endif

#include <Eigen/Core>
#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

#include "panna/data.hpp"
#include "panna/distance.hpp"

namespace panna {

    //! Whether `PairForestIndex` may evaluate its tiles of `Dataset` points
    //! under `Distance` with the GEMM filter. Disabled unless specialised.
    template <typename Dataset, typename Distance>
    struct GemmTileKernel {
        static constexpr bool enabled = false;
    };

    template <>
    struct GemmTileKernel<EuclideanPoints, EuclideanDistance> {
        static constexpr bool enabled = true;
        //! `false`: the metric is the square root of the squared distance.
        static constexpr bool squared = false;
    };

    template <>
    struct GemmTileKernel<EuclideanPoints, EuclideanDistanceNoSqrt> {
        static constexpr bool enabled = true;
        //! `true`: the metric *is* the squared distance.
        static constexpr bool squared = true;
    };

    //! When `PairForestIndex` evaluates a tile with the GEMM filter.
    //!
    //! The filter only pays off when it rejects enough pairs: the GEMM costs
    //! roughly half a scalar distance per pair of the tile (measured at d = 20,
    //! 128 and 784), and every pair it fails to reject still costs a full scalar
    //! distance on top. In the EMST search the hash family is fitted to the
    //! current cutoff, so most colliding pairs *pass* the threshold and the
    //! filter has little to do -- `adaptive` notices that and stays scalar.
    enum class GemmPolicy {
        never,    //!< always the scalar kernel
        adaptive, //!< the GEMM filter while it rejects enough pairs (the default)
        always,   //!< the GEMM filter on every tile, however small (for tests)
    };

    //! Decides, from the GEMM block, whether a pair certainly fails
    //! `Distance::compute( a, b ) <= threshold`. `squared` is
    //! `GemmTileKernel<Dataset, Distance>::squared`.
    //!
    //! The GEMM does not see the points themselves but the points translated by
    //! a per-tile centre `m` (see `GemmTileBuffers`), which leaves every distance
    //! unchanged and shrinks the norms the rounding error scales with: without
    //! it, data far from the origin (every coordinate `+1000`, say) would have
    //! an error bound far above its distances and nothing would be rejected.
    //!
    //! Error model, with `u = FLT_EPSILON / 2` and `S = |a - b|^2` in exact
    //! arithmetic. A float dot product of length `d`, whatever its summation
    //! order or use of FMA, is off by at most `gamma_d * sum |x_i y_i|`, with
    //! `gamma_d = d u / (1 - d u)` and `sum |x_i y_i| <= (|x|^2 + |y|^2) / 2`.
    //!  - Estimate. With `N' = |a'|^2 + |b'|^2` for the *translated* points
    //!    `a' = fl( a - m )`, `E = N' - 2 a'.b'` is off from `S` by at most
    //!    `~2 (d + 3) u N'` for the GEMM and the norms, plus `~4 u N'` for the
    //!    rounding of the translation itself.
    //!  - `EuclideanDistance::compute` sums squared *differences*, so it is off
    //!    from `S` only by a relative `~(d + 3) u`, `sqrt` included. It needs no
    //!    term of its own: a rejected pair has `thr^2 < S <= ~2 N'`, so that
    //!    error is below `~2 (d + 3) u N'`, which the slack's margin covers.
    //!  - `EuclideanDistanceNoSqrt::compute` evaluates `|a|^2 + |b|^2 - 2 a.b` on
    //!    the *untranslated* points, so it is off from `S` by up to
    //!    `~(2 d + 3) u N`, `N = |a|^2 + |b|^2`. That error is part of the value
    //!    the scalar kernel compares with the threshold, so the slack must cover
    //!    it too, whatever the translation gains for the estimate: on data far
    //!    from the origin this metric inherently leaves little to reject.
    //! Each slack term below is `8 (d + 8) u` times its norm, over twice the
    //! error it covers, which also absorbs the handful of roundings made while
    //! evaluating the test itself. The absolute term covers underflow, where a
    //! product may lose up to `2^-150` whatever its magnitude.
    //!
    //! A pair is rejected only when `E - slack > threshold` holds; any NaN on the
    //! left (a NaN or infinite coordinate, an overflowing norm) makes the test
    //! false, so such pairs are handed to `Distance::compute` like any other.
    template <bool squared>
    class GemmRejection {
    public:
        //! `distance_threshold` is in the units of `Distance`.
        GemmRejection( float distance_threshold, size_t dimensions ):
            relative_slack( 4.0f * static_cast<float>( dimensions + 8 ) * FLT_EPSILON ),
            absolute_slack( 4.0f * static_cast<float>( dimensions + 8 ) * FLT_MIN ) {
            constexpr float inf = std::numeric_limits<float>::infinity();
            if ( std::isnan( distance_threshold ) ) {
                /// `distance <= NaN` never holds, so every pair fails: reject all
                /// of them (a NaN estimate still goes to the exact test, and
                /// fails there).
                threshold = -inf;
            } else if constexpr ( squared ) {
                /// Negative thresholds are kept as they are: the scalar formula
                /// can itself round below zero, and the slack covers that.
                threshold = distance_threshold;
            } else if ( distance_threshold >= 0.0f ) {
                /// `-0.0f` lands here too, and must: `sqrt( 0 ) <= -0.0f` holds.
                /// A square that overflows gives `+inf`, i.e. "reject nothing".
                threshold = distance_threshold * distance_threshold;
            } else {
                /// A square root is never negative: nothing can pass.
                threshold = -inf;
            }
        }

        //! `true` when no pair can ever be rejected (an infinite threshold), in
        //! which case the GEMM would only be overhead.
        bool rejects_nothing() const {
            return threshold == std::numeric_limits<float>::infinity();
        }

        //! `true` when the pair certainly fails the threshold. `norm_a`, `norm_b`
        //! are the squared norms of the translated points and `dot` their GEMM
        //! inner product; `exact_norm_a`, `exact_norm_b` are the squared norms of
        //! the points themselves, only needed when `squared`.
        bool rejects( float norm_a,
                      float norm_b,
                      float dot,
                      [[maybe_unused]] float exact_norm_a,
                      [[maybe_unused]] float exact_norm_b ) const {
            const float norms = norm_a + norm_b;
            float slack = relative_slack * norms + absolute_slack;
            if constexpr ( squared ) {
                slack += relative_slack * ( exact_norm_a + exact_norm_b );
            }
            return norms - 2.0f * dot - slack > threshold;
        }

    private:
        float relative_slack;
        float absolute_slack;
        float threshold; //!< in squared-distance units, inflated
    };

    //! Per-thread scratch of the GEMM tile kernel: the gathered row and column
    //! points, their inner-product block and their squared norms.
    //!
    //! Every chunk of coordinates is translated by the mean of the tile's row
    //! points before the product (see `GemmRejection` for why), so the block
    //! holds `(a - m).(b - m)` and the norms are those of `a - m`.
    //!
    //! The dimensions are processed in chunks of at most `DIM_CHUNK` floats, the
    //! products of the chunks accumulating into `block`. That bounds the scratch
    //! at `2 * 128 * DIM_CHUNK + 128 * 128` floats (320 KiB) per thread however
    //! wide the points are, and keeps each gathered chunk cache-resident.
    //! Storage is grown on demand and never shrunk, so after the first tile of
    //! a search no further allocation takes place.
    //!
    //! It also keeps the statistics behind `GemmPolicy::adaptive`: a running
    //! average of the fraction of each GEMM block that the filter rejected.
    class GemmTileBuffers {
    public:
        using Matrix = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

        static constexpr size_t DIM_CHUNK = 256;

        //! Below this fraction of rejected pairs per pair of the block, the
        //! GEMM costs more than the scalar distances it saves.
        static constexpr float MIN_REJECTED_FRACTION = 0.5f;

        //! While the filter does not pay, one tile in this many still goes
        //! through it, so that the running average can notice when a lower
        //! threshold makes it pay again. That bounds the overhead of a useless
        //! filter at `MIN_REJECTED_FRACTION / PROBE_INTERVAL` scalar distances
        //! per pair (about 3%).
        static constexpr uint32_t PROBE_INTERVAL = 16;

        //! Whether `GemmPolicy::adaptive` should filter the next tile.
        bool worth_filtering() {
            if ( rejected_fraction >= MIN_REJECTED_FRACTION ) {
                return true;
            }
            if ( ++tiles_since_probe >= PROBE_INTERVAL ) {
                tiles_since_probe = 0;
                return true;
            }
            return false;
        }

        //! Records that a filtered tile rejected `rejected` of its `pairs`.
        void record( size_t rejected, size_t pairs ) {
            if ( pairs > 0 ) {
                const float fraction = static_cast<float>( rejected ) / static_cast<float>( pairs );
                rejected_fraction = 0.5f * ( rejected_fraction + fraction );
            }
        }

        //! Fills `products_row( i )[j]` with the inner product of the translated
        //! points `row_ids[i]` and `col_ids[j]`, and the squared norms. `same_ids`
        //! says that the two id ranges coincide (a diagonal tile), so the points
        //! are gathered once and the column norms alias the row norms.
        //! `exact_norms` also fills the norms of the untranslated points.
        void compute( const EuclideanPoints& points,
                      const uint32_t* row_ids,
                      size_t num_rows,
                      const uint32_t* col_ids,
                      size_t num_cols,
                      bool same_ids,
                      bool exact_norms ) {
            const size_t dimensions = points.get_dimensions();
            const size_t width = std::min( dimensions, DIM_CHUNK );
            grow( rows, num_rows, width );
            grow( block, num_rows, num_cols );
            if ( centre.size() < static_cast<Eigen::Index>( width ) ) {
                centre.resize( static_cast<Eigen::Index>( width ) );
            }
            row_side.reset( num_rows, exact_norms );
            if ( !same_ids ) {
                grow( cols, num_cols, width );
                col_side.reset( num_cols, exact_norms );
            }
            aliased = same_ids;

            auto products = block.topLeftCorner( num_rows, num_cols );
            for ( size_t offset = 0; offset < dimensions; offset += DIM_CHUNK ) {
                const size_t chunk = std::min( DIM_CHUNK, dimensions - offset );
                auto r = rows.topLeftCorner( num_rows, chunk );
                gather( points, row_ids, offset, r, row_side, exact_norms );
                /// Any centre keeps the distances; the mean of the rows is cheap
                /// and sits inside the tile's cluster of points.
                auto m = centre.head( static_cast<Eigen::Index>( chunk ) );
                m = r.colwise().mean();
                translate( r, m, row_side );
                if ( same_ids ) {
                    if ( offset == 0 ) {
                        products.noalias() = r * r.transpose();
                    } else {
                        products.noalias() += r * r.transpose();
                    }
                } else {
                    auto c = cols.topLeftCorner( num_cols, chunk );
                    gather( points, col_ids, offset, c, col_side, exact_norms );
                    translate( c, m, col_side );
                    if ( offset == 0 ) {
                        products.noalias() = r * c.transpose();
                    } else {
                        products.noalias() += r * c.transpose();
                    }
                }
            }
        }

        //! Row `i` of the inner-product block, contiguous over the columns.
        const float* products_row( size_t i ) const {
            return block.data() + i * static_cast<size_t>( block.cols() );
        }

        //! Squared norms of the translated row points.
        const float* row_norms() const {
            return row_side.norms.data();
        }

        //! Squared norms of the translated column points.
        const float* col_norms() const {
            return aliased ? row_side.norms.data() : col_side.norms.data();
        }

        //! Squared norms of the row points themselves (if `exact_norms`).
        const float* row_exact_norms() const {
            return row_side.exact_norms.data();
        }

        //! Squared norms of the column points themselves (if `exact_norms`).
        const float* col_exact_norms() const {
            return aliased ? row_side.exact_norms.data() : col_side.exact_norms.data();
        }

    private:
        //! The norms of one side (rows or columns) of the tile, summed over the
        //! chunks.
        struct Side {
            std::vector<float> norms;
            std::vector<float> exact_norms;

            void reset( size_t count, bool exact ) {
                norms.assign( count, 0.0f );
                if ( exact ) {
                    exact_norms.assign( count, 0.0f );
                }
            }
        };

        using Block = Eigen::Block<Matrix>;

        Matrix rows;               //!< gathered chunk of the row points
        Matrix cols;               //!< gathered chunk of the column points (off-diagonal tiles)
        Matrix block;              //!< the inner products, row-major
        Eigen::RowVectorXf centre; //!< the translation, `DIM_CHUNK` coordinates at a time
        Side row_side;
        Side col_side;
        bool aliased = false; //!< whether the columns are the rows
        //! Running average behind `worth_filtering`. Optimistic at first, so the
        //! first eligible tile tries the filter.
        float rejected_fraction = 1.0f;
        uint32_t tiles_since_probe = 0;

        //! Makes `m` at least `num_rows x num_cols`, keeping it if it already is.
        //! The contents are discarded, which is fine: they are always rewritten.
        static void grow( Matrix& m, size_t num_rows, size_t num_cols ) {
            const Eigen::Index r = static_cast<Eigen::Index>( num_rows );
            const Eigen::Index c = static_cast<Eigen::Index>( num_cols );
            if ( m.rows() < r || m.cols() < c ) {
                m.resize( std::max( m.rows(), r ), std::max( m.cols(), c ) );
            }
        }

        //! Copies coordinates `[offset, offset + out.cols())` of the points `ids`
        //! into the rows of `out`, adding their squares to `side.exact_norms`
        //! when asked to.
        static void gather( const EuclideanPoints& points,
                            const uint32_t* ids,
                            size_t offset,
                            Block& out,
                            Side& side,
                            bool exact_norms ) {
            const size_t chunk = static_cast<size_t>( out.cols() );
            for ( Eigen::Index i = 0; i < out.rows(); i++ ) {
                float* dst = &out( i, 0 );
                std::memcpy( dst, points[ids[i]].vector + offset, chunk * sizeof( float ) );
                if ( exact_norms ) {
                    side.exact_norms[i] += out.row( i ).squaredNorm();
                }
            }
        }

        //! Subtracts `m` from every row of `out`, adding the squares of the
        //! result to `side.norms`.
        template <typename Centre>
        static void translate( Block& out, const Centre& m, Side& side ) {
            for ( Eigen::Index i = 0; i < out.rows(); i++ ) {
                out.row( i ) -= m;
                side.norms[i] += out.row( i ).squaredNorm();
            }
        }
    };

    //! Stand-in for `GemmTileBuffers` in the scratch of a `(Dataset, Distance)`
    //! pair that keeps the scalar kernel, so that its scratch stays as small as
    //! it was.
    struct NoGemmTileBuffers {};

} // namespace panna
