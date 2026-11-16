---
title: Faster index construction for CrossPolytope and Simhash
status: proposed
created: 2026-11-16
revised: 2026-11-16 (detailed implementation steps; code verified in a scratch worktree)
---

# Faster index construction for CrossPolytope and Simhash

## Goal

Make `PairForestIndex` construction (the `index_ms` phase of
`pair_forest_emst`) under `CrossPolytope` and `Simhash` at least as fast as
under `LatticeLSH`. Today it takes about 40 s with CrossPolytope, against
about 5 s with LatticeLSH, on 100k glove points. CP and Simhash already
compute fewer distances in the discovery phase, so once construction is fast
they should win end to end.

## Decisions (agreed with the user)

| Topic | Decision |
|---|---|
| Scope | Only the LSH implementation changes: `include/panna/lsh/simhash.hpp`, `include/panna/lsh/crosspolytope.hpp`, `include/panna/linalg.hpp` (`RandomDotProducts`), plus tests, a benchmark, docstrings and the CHANGELOG. |
| Off limits | `include/panna/pairemst.hpp`, `include/panna/pairforest.hpp`, `include/panna/prefixmap.hpp`, `include/panna/trieindex.hpp`, `include/panna/emst.hpp`. Do not edit them, not even their comments. |
| Interface | `hash(point, output)` stays per point. `pairforest.hpp` calls it once per point from an OpenMP loop. |
| CrossPolytope rounds | **Fixed at 3** independent (random ±1 diagonal, FHT) rounds per symbol. Do **not** add fewer rounds, shared rounds, or a "rounds" parameter. |
| Simhash | Compute all projections with one `RandomDotProducts`, like `LatticeLSH`. |
| Defaults | `pairemst.hpp` builds the hash builders as `Builder( dimensions )`, so every improvement must come from defaults inside the LSH code. |

## Expected outcome (measured on the reference implementation in this plan)

The code below was implemented and tested in a scratch git worktree on
2026-11-16 (base commit `6b1974f`). Numbers: glove-100-angular, 100k-point
sample, L = 512, K as in `EMST_exposed` (CP 4, Simhash 24, Lattice 4), 16
threads on an i7-1260P.

| | before | after |
|---|---|---|
| CP hasher construction (`build`) | 2,377 ms | 149 ms |
| CP hashing, all points, 16 threads | 36,185 ms | 14,293 ms |
| CP hashing, 1 pinned thread | 754 µs/pt | 439 µs/pt |
| Simhash hasher construction | 240 ms | 0.5 ms |
| Simhash hashing, all points, 16 threads | 3,337 ms | 1,340 ms |
| Simhash hashing, 1 pinned thread | 72 µs/pt | 35 µs/pt |
| `pair_forest_emst` index_ms, CP (eps 0.1) | 40,176 | 15,877 |
| `pair_forest_emst` index_ms, Simhash (eps 0.1) | 4,545 | 2,427 |
| `pair_forest_emst` total, CP / Simhash / Lattice (eps 0.1) | 61.6 / 42.4 / 50.3 s | 34.2 / 40.2 / 38.3 s |

Tree weights stayed within the same 0.03% band. All C++ and Python tests
pass, except the pre-existing failures listed below.

---

## Instructions for the implementer

Read this whole section before starting.

1. **Work on the branch `fast-cp-simhash-hashing`** (it already exists and
   contains this plan). Make **one commit per step**, with the message given
   in the step. Each step must leave all tests passing (apart from the
   pre-existing failures below).
2. **Every build, test and benchmark command must run inside the nix
   devshell**, i.e. prefixed with `nix develop -c`. Eigen, Catch2, HighFive
   and the C++ runtime that NumPy links against are only available there.
   Without it, `pip install .` fails with "Could not find a package
   configuration file provided by Eigen3", and `import numpy` fails with
   "libstdc++.so.6: cannot open shared object file".
3. **Copy the code blocks of this plan exactly.** They were compiled and
   tested as written. If something does not compile, check first that you
   copied the whole block and the right file.
4. Do **not** change anything that is not listed in a step. In particular,
   do not touch the files that are off limits (see Decisions).
5. If a check in a step does not give the expected result, stop and report
   what you saw, instead of changing tests or tolerances that the step does
   not mention.

### Commands

```sh
# C++: configure once (only if build/cpp/CMakeCache.txt does not exist)
nix develop -c cmake -S . -B build/cpp -DCMAKE_BUILD_TYPE=RelWithDebInfo

# C++: build the tests and the benchmark (re-run cmake -S . -B build/cpp after editing CMakeLists.txt)
nix develop -c cmake --build build/cpp -j --target tests bench-lsh-build

# C++: run a subset of the tests (Catch2 test-name or [tag] filters, comma = OR)
nix develop -c build/cpp/tests "[lsh-quality]"
nix develop -c build/cpp/tests "*Simhash*,*CrossPolytope*,*ensor*,Failure probability*"
nix develop -c build/cpp/tests "[pairforest]"
nix develop -c build/cpp/tests "*trie*"
nix develop -c build/cpp/tests "[pairemst]"

# C++: the full suite (takes a few minutes)
nix develop -c build/cpp/tests

# Python: rebuild and install the extension into .venv (a few minutes), then test
nix develop -c .venv/bin/pip install .
nix develop -c .venv/bin/python -m pytest -q python/tests/test_lsh.py
nix develop -c .venv/bin/python -m pytest -q python/tests
```

### Pre-existing test failures (at commit `6b1974f`, before any change)

Running `build/cpp/tests` on unmodified code gives `2 failed`:

- `test/include/linalg_test.hpp:115`, `REQUIRE( std::abs(input_norm - output_norm) <= 1e-5 )`
  (875.99 <= 0.00001). This always fails and is unrelated to this work.
- `test/include/emst_test.hpp:87`, `REQUIRE( exact.first == probabilistic_weight )`.
  This one is flaky: it fails on some runs and not on others.

Record the output of the full suite **before step 1**, and compare against
it after each step. These two may fail, nothing else. The Python suite
`python/tests/test_lsh.py` passes fully (127 tests).

---

## Step 1: build benchmark (`bench-lsh-build`)

**Goal:** a reproducible measurement of the hashing work that
`PairForestIndex::build_all_repetitions` does. It covers hasher
construction, hashing every point with all threads, and the per-repetition
sort, and it also has a single-thread mode.

**Files:** new `test/bench_lsh_build.cpp`; `CMakeLists.txt`.

1. Create `test/bench_lsh_build.cpp` with exactly this content:

```cpp
// Benchmark of the hashing work done when `PairForestIndex` is constructed
// (see `PairForestIndex::build_all_repetitions`), for the cosine families
// used by `pair_forest_emst` and, for reference, `LatticeLSH`, with the K and
// L of `EMST_exposed` in `python/wrapper.cpp`.
//
//   bench-lsh-build <dataset.hdf5> [num_points=100000] [family=all] [mode=parallel]
//
// family: all | lattice | crosspolytope | simhash
// mode:
//   parallel  times (best of 3) the construction of the hasher, the hashing of
//             every point with all OpenMP threads (scattering the values per
//             repetition, as the index does) and the per-repetition sort.
//   single    hashes the first 5000 points on the calling thread only; run it
//             pinned under `perf stat -e cycles:u` (on hybrid Intel CPUs,
//             `cpu_core/cycles/u` with `taskset -c <P-core>`) for a stable,
//             turbo-independent cost per point.
//
// The points are a random sample (seed 1234) of the "/train" dataset,
// normalized to unit length. LatticeLSH is not fitted (zero offset, fixed
// scaling factor): its timings are a reference only.

#include <highfive/H5Easy.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "panna/data.hpp"
#include "panna/distance.hpp"
#include "panna/lsh/crosspolytope.hpp"
#include "panna/lsh/lattice.hpp"
#include "panna/lsh/simhash.hpp"
#include "panna/rand.hpp"

using namespace panna;
using Clock = std::chrono::steady_clock;

constexpr size_t REPETITIONS = 512;
constexpr size_t SINGLE_MODE_POINTS = 5000;

static double ms_since( Clock::time_point start ) {
    return std::chrono::duration<double, std::milli>( Clock::now() - start ).count();
}

template <typename Dataset, typename Make>
void run( const std::string& name, const Dataset& data, Make make, const std::string& mode ) {
    using Hasher = decltype( make() );
    using Value = typename Hasher::Value;
    const size_t n = data.size();

    auto start = Clock::now();
    Hasher hasher = make();
    const double build_ms = ms_since( start );

    if ( mode == "single" ) {
        std::vector<Value> values;
        size_t checksum = 0;
        start = Clock::now();
        for ( size_t i = 0; i < std::min( n, SINGLE_MODE_POINTS ); i++ ) {
            hasher.hash( data[i], values );
            checksum += values[0] == values[1];
        }
        std::printf( "%-14s single: %.2f us/point (checksum %zu)\n", name.c_str(),
                     ms_since( start ) * 1e3 / std::min( n, SINGLE_MODE_POINTS ), checksum );
        return;
    }

    std::vector<std::vector<Value>> per_repetition( REPETITIONS, std::vector<Value>( n ) );
    double hash_ms = 1e300;
    for ( int trial = 0; trial < 3; trial++ ) {
        start = Clock::now();
#pragma omp parallel
        {
            std::vector<Value> values;
#pragma omp for schedule( static )
            for ( size_t i = 0; i < n; i++ ) {
                hasher.hash( data[i], values );
                for ( size_t rep = 0; rep < REPETITIONS; rep++ ) {
                    per_repetition[rep][i] = values[rep];
                }
            }
        }
        hash_ms = std::min( hash_ms, ms_since( start ) );
    }

    start = Clock::now();
#pragma omp parallel for schedule( dynamic, 1 )
    for ( size_t rep = 0; rep < REPETITIONS; rep++ ) {
        std::vector<std::pair<Value, uint32_t>> pairs;
        pairs.reserve( n );
        for ( size_t i = 0; i < n; i++ ) {
            pairs.emplace_back( per_repetition[rep][i], static_cast<uint32_t>( i ) );
        }
        std::sort( pairs.begin(), pairs.end() );
    }
    const double sort_ms = ms_since( start );

    std::printf( "%-14s build %8.1f ms | hash %9.1f ms (%6.2f us/point) | sort %8.1f ms\n",
                 name.c_str(), build_ms, hash_ms, hash_ms * 1e3 / n, sort_ms );
}

int main( int argc, char** argv ) {
    if ( argc < 2 ) {
        std::cerr << "usage: " << argv[0]
                  << " <dataset.hdf5> [num_points=100000] [family=all] [mode=parallel]\n";
        return 1;
    }
    const std::string path = argv[1];
    const size_t num_points = argc > 2 ? std::stoul( argv[2] ) : 100000;
    const std::string family = argc > 3 ? argv[3] : "all";
    const std::string mode = argc > 4 ? argv[4] : "parallel";

    H5Easy::File file( path, H5Easy::File::ReadOnly );
    auto rows = H5Easy::load<std::vector<std::vector<float>>>( file, "/train" );
    std::vector<size_t> order( rows.size() );
    std::iota( order.begin(), order.end(), 0 );
    std::mt19937_64 shuffle_rng( 1234 );
    std::shuffle( order.begin(), order.end(), shuffle_rng );
    order.resize( std::min( num_points, rows.size() ) );

    const size_t d = rows[0].size();
    UnitNormPoints unit( d );
    EuclideanPoints euclidean( d );
    for ( size_t idx : order ) {
        std::vector<float> row = rows[idx];
        double norm = 0;
        for ( float v : row ) {
            norm += v * v;
        }
        norm = std::sqrt( norm );
        if ( norm > 0 ) {
            for ( float& v : row ) {
                v /= norm;
            }
        }
        unit.push_back( row.begin(), row.end() );
        euclidean.push_back( row.begin(), row.end() );
    }
    rows.clear();
    std::printf( "%zu points, %zu dimensions, L=%zu, %d threads, mode %s\n", unit.size(), d,
                 REPETITIONS, omp_get_max_threads(), mode.c_str() );

    seed_global_rng( 1234 );
    if ( family == "all" || family == "lattice" ) {
        run( "lattice K=4", euclidean, [&] {
            return LatticeLSH<4, EuclideanPoints, EuclideanDistance>(
                std::vector<float>( d, 0.0f ), 0.92f, d, REPETITIONS );
        }, mode );
    }
    if ( family == "all" || family == "crosspolytope" ) {
        run( "crosspolytope K=4", unit, [&] {
            return CrossPolytopeBuilder<4, UnitNormPoints, CosineDistance>( d ).build( REPETITIONS );
        }, mode );
    }
    if ( family == "all" || family == "simhash" ) {
        run( "simhash K=24", unit, [&] {
            return SimhashBuilder<24, UnitNormPoints, CosineDistance>( d ).build( REPETITIONS );
        }, mode );
    }
    return 0;
}
```

2. In `CMakeLists.txt`, find these lines (inside `if (NOT SKBUILD)`):

```cmake
  add_executable(bench test/bench.cpp)
  add_executable(bench_lattice_dots test/bench_lattice_dots.cpp)
```

   and add right after them:

```cmake

  add_executable(bench-lsh-build test/bench_lsh_build.cpp)
  target_link_libraries(bench-lsh-build PRIVATE HighFive)
```

3. Build and run:

```sh
nix develop -c cmake -S . -B build/cpp -DCMAKE_BUILD_TYPE=RelWithDebInfo
nix develop -c cmake --build build/cpp -j --target bench-lsh-build
nix develop -c build/cpp/bench-lsh-build datasets/glove-100-angular.hdf5 100000 all parallel
for f in lattice crosspolytope simhash; do
  for t in 1 2 3; do
    OMP_NUM_THREADS=1 nix develop -c taskset -c 2 build/cpp/bench-lsh-build datasets/glove-100-angular.hdf5 6000 $f single
  done
done
```

**Expected** (before any library change; timings vary by up to 2× between
runs on a laptop, so take the best of the three single-thread runs):

```
lattice K=4    build      2.8 ms | hash    2289.1 ms ( 22.89 us/point) | sort   1508.6 ms
crosspolytope K=4 build   2376.5 ms | hash   36185.0 ms (361.85 us/point) | sort    968.9 ms
simhash K=24   build    240.2 ms | hash    3337.4 ms ( 33.37 us/point) | sort    653.3 ms
```

Single thread (best of 3): crosspolytope about 754 µs/point, simhash about
72 µs/point, lattice about 42 µs/point. Lattice is bimodal (about 41 or
about 60 µs/point) from run to run.

Save these numbers: step 6 needs them.

**Commit:** `add a benchmark of the hashing done by the pair forest index construction`

---

## Step 2: collision-quality test (`[lsh-quality]`)

**Goal:** a test that checks the statistical properties the index relies on,
for Simhash and CrossPolytope. The existing `test_hash_collision_probability`
only averages over random, nearly orthogonal pairs. On pairs at controlled
dot products `alpha` ∈ {0.95, 0.9, 0.8, 0.6} it checks that:
(1) the fraction of colliding symbols matches the single-symbol probability
`p`; (2) the fraction of colliding repetitions matches `p^K`; (3) over 500
pairs, the variance of the number of colliding repetitions out of L = 512 is
binomial (ratio in [0.75, 1.3]), i.e. the repetitions are independent.

**Files:** new `test/include/lsh/quality_test.hpp`; `test/tests.cpp`.

1. Create `test/include/lsh/quality_test.hpp` with exactly this content:

```cpp
#pragma once

// Statistical quality checks for the cosine LSH families (`Simhash`,
// `CrossPolytope`): on pairs of points at a controlled dot product `alpha`,
// the empirical collision rates must match what the index assumes.
//
//  1. per symbol: the fraction of colliding symbols (bits for Simhash) is the
//     single-symbol collision probability `p(alpha)`;
//  2. per repetition: the fraction of colliding repetitions (all `K` symbols
//     equal) is `p^K`, i.e. the symbols of a repetition are independent;
//  3. across repetitions: over many pairs, the number of colliding
//     repetitions out of `L` has the variance of a Binomial(L, p^K), i.e. the
//     repetitions are independent. The index's `failure_probability` relies
//     on this; a shared source of randomness would show up as excess
//     variance (ratio > 1).

#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <random>
#include <vector>

#include "panna/data.hpp"
#include "panna/distance.hpp"
#include "panna/linalg.hpp"
#include "panna/lsh/crosspolytope.hpp"
#include "panna/lsh/simhash.hpp"
#include "panna/rand.hpp"

namespace panna {
    namespace lsh_quality {

        constexpr size_t DIMENSIONS = 100;
        constexpr size_t REPETITIONS = 512;
        constexpr size_t NUM_PAIRS = 500;
        constexpr float ALPHAS[] = { 0.95f, 0.9f, 0.8f, 0.6f };

        struct Rates {
            double symbol = 0;    // fraction of colliding symbols
            double full = 0;      // fraction of colliding repetitions
            double var_ratio = 0; // var(#colliding repetitions per pair) / binomial variance
        };

        //! Number of equal symbols of two values.
        template <uint8_t K>
        size_t equal_symbols( const BitwiseLshValue<K>& a, const BitwiseLshValue<K>& b ) {
            const uint32_t mask = K == 32 ? 0xffffffffu : ( ( 1u << K ) - 1 );
            return K - __builtin_popcount( ( a.get_bits() ^ b.get_bits() ) & mask );
        }

        template <typename Symbol, uint8_t K>
        size_t equal_symbols( const SymbolLshValue<Symbol, K>& a, const SymbolLshValue<Symbol, K>& b ) {
            size_t equal = 0;
            for ( size_t c = 0; c < K; c++ ) {
                equal += a.get( c ) == b.get( c );
            }
            return equal;
        }

        //! A pair of unit vectors with dot product `alpha`: `x` is a random
        //! direction and `y = alpha x + sqrt(1 - alpha^2) u`, with `u` a random
        //! direction orthogonal to `x`.
        inline void make_pair( float alpha, std::mt19937_64& rng, std::vector<float>& x,
                               std::vector<float>& y ) {
            std::normal_distribution<float> normal( 0.0f, 1.0f );
            std::vector<float> u( DIMENSIONS );
            x.resize( DIMENSIONS );
            y.resize( DIMENSIONS );
            double xx = 0;
            for ( size_t i = 0; i < DIMENSIONS; i++ ) {
                x[i] = normal( rng );
                u[i] = normal( rng );
                xx += x[i] * x[i];
            }
            double xu = 0;
            for ( size_t i = 0; i < DIMENSIONS; i++ ) {
                x[i] /= std::sqrt( xx );
                xu += x[i] * u[i];
            }
            double uu = 0;
            for ( size_t i = 0; i < DIMENSIONS; i++ ) {
                u[i] -= xu * x[i];
                uu += u[i] * u[i];
            }
            const float beta = std::sqrt( 1.0f - alpha * alpha );
            for ( size_t i = 0; i < DIMENSIONS; i++ ) {
                y[i] = alpha * x[i] + beta * u[i] / std::sqrt( uu );
            }
        }

        template <typename Hasher>
        Rates measure( Hasher& hasher, float alpha ) {
            using Value = typename Hasher::Value;
            constexpr size_t K = Hasher::get_concatenations();
            std::mt19937_64 rng( 777 );
            std::vector<float> x, y;
            std::vector<Value> hx, hy;
            std::vector<double> counts;
            double symbols = 0;
            double full = 0;
            for ( size_t p = 0; p < NUM_PAIRS; p++ ) {
                make_pair( alpha, rng, x, y );
                UnitNormPoints points( DIMENSIONS );
                points.push_back( x.begin(), x.end() );
                points.push_back( y.begin(), y.end() );
                hasher.hash( points[0], hx );
                hasher.hash( points[1], hy );
                double count = 0;
                for ( size_t rep = 0; rep < REPETITIONS; rep++ ) {
                    const size_t equal = equal_symbols( hx[rep], hy[rep] );
                    symbols += equal;
                    count += equal == K;
                }
                full += count;
                counts.push_back( count );
            }
            Rates rates;
            rates.symbol = symbols / ( NUM_PAIRS * REPETITIONS * K );
            rates.full = full / ( NUM_PAIRS * REPETITIONS );
            const double mean = full / NUM_PAIRS;
            double var = 0;
            for ( double c : counts ) {
                var += ( c - mean ) * ( c - mean );
            }
            var /= NUM_PAIRS - 1;
            rates.var_ratio = var / ( REPETITIONS * rates.full * ( 1 - rates.full ) );
            return rates;
        }

        //! Reference single-symbol collision probability of the cross-polytope
        //! hash in `padded` dimensions at dot product `alpha`, with a Gaussian
        //! matrix, from `samples` Monte Carlo samples (standard error
        //! `<= 0.5 / sqrt(samples)`). Independent of
        //! `CrossPolytopeCollisionEstimates`, whose default table is too noisy
        //! for these checks.
        inline double reference_cp_probability( size_t padded, float alpha, size_t samples ) {
            std::mt19937_64 rng( 2024 );
            std::normal_distribution<float> normal( 0.0f, 1.0f );
            const float beta = std::sqrt( 1.0f - alpha * alpha );
            size_t collisions = 0;
            for ( size_t s = 0; s < samples; s++ ) {
                size_t hx = 0, hy = 0;
                float bx = -1, by = -1;
                for ( size_t j = 0; j < padded; j++ ) {
                    const float z1 = normal( rng );
                    const float z2 = normal( rng );
                    const float vy = alpha * z1 + beta * z2;
                    if ( std::fabs( z1 ) > bx ) {
                        bx = std::fabs( z1 );
                        hx = z1 < 0 ? j + padded : j;
                    }
                    if ( std::fabs( vy ) > by ) {
                        by = std::fabs( vy );
                        hy = vy < 0 ? j + padded : j;
                    }
                }
                collisions += hx == hy;
            }
            return static_cast<double>( collisions ) / samples;
        }

        //! Checks the three rates against the single-symbol probability `p`.
        inline void check_rates( const Rates& r, double p, size_t K ) {
            const double pk = std::pow( p, K );
            INFO( "symbol rate " << r.symbol << " vs p " << p << "; full rate " << r.full
                                 << " vs p^K " << pk << "; variance ratio " << r.var_ratio );
            CHECK( std::abs( r.symbol - p ) <= 0.006 );
            const double full_sd = std::sqrt( pk * ( 1 - pk ) / ( NUM_PAIRS * REPETITIONS ) );
            CHECK( std::abs( r.full - pk ) <= 0.1 * pk + 4 * full_sd );
            // The variance ratio is meaningful only with enough expected
            // collisions per pair; its standard error is about sqrt(2 / NUM_PAIRS).
            if ( REPETITIONS * pk >= 5 ) {
                CHECK( r.var_ratio >= 0.75 );
                CHECK( r.var_ratio <= 1.3 );
            }
        }

    } // namespace lsh_quality

    TEST_CASE( "Simhash collision quality", "[lsh-quality]" ) {
        using namespace lsh_quality;
        seed_global_rng( 11 );
        Simhash<24, UnitNormPoints, CosineDistance> hasher( DIMENSIONS, REPETITIONS );
        for ( float alpha : ALPHAS ) {
            INFO( "alpha = " << alpha );
            const double p = 1.0 - std::acos( alpha ) / M_PI;
            check_rates( measure( hasher, alpha ), p, 24 );
        }
    }

    TEST_CASE( "CrossPolytope collision quality", "[lsh-quality][!mayfail]" ) {
        using namespace lsh_quality;
        seed_global_rng( 12 );
        using Hasher = CrossPolytope<4, UnitNormPoints, CosineDistance>;
        Hasher hasher = CrossPolytopeBuilder<4, UnitNormPoints, CosineDistance>( DIMENSIONS )
                            .build( REPETITIONS );
        for ( float alpha : ALPHAS ) {
            INFO( "alpha = " << alpha );
            const double p = reference_cp_probability( 128, alpha, 200000 );
            check_rates( measure( hasher, alpha ), p, 4 );
        }
    }

} // namespace panna
```

   In this step, the CrossPolytope test case is tagged `[!mayfail]`,
   because the **current** CrossPolytope fails check (1). Step 4 removes
   the tag.

2. In `test/tests.cpp`, after the line `#include "lsh/hash_test.hpp"`, add:

```cpp
#include "lsh/quality_test.hpp"
```

3. Build and run:

```sh
nix develop -c cmake --build build/cpp -j --target tests
nix develop -c build/cpp/tests "[lsh-quality]"
```

**Expected:** about 20 s. Simhash passes. CrossPolytope reports 3 failed
checks, but the run as a whole succeeds because of `[!mayfail]`. The
failures look like:

```
quality_test.hpp:167: FAILED:
  alpha = 0.95
  symbol rate 0.578692 vs p 0.571445; full rate 0.111844 vs p^K 0.106635; ...
  alpha = 0.9
  symbol rate 0.441396 vs p 0.432485; ...
  alpha = 0.8
  symbol rate 0.281987 vs p 0.27365; ...
```

This is a real bug of the current CrossPolytope, which step 4 fixes.
`encode_closest_axis` only scans the first `dimensions` (= 100) of the 128
rotated coordinates, so it behaves like a 100-axis cross-polytope, while its
collision probabilities are estimated for 128 axes. A Gaussian simulation
gives 0.579 / 0.443 / 0.284 with 100 axes and 0.571 / 0.432 / 0.274 with
128, which matches the observed and predicted rates. The index therefore
currently overestimates the failure probability, which is conservative but
wrong.

**Commit:** `add statistical collision-quality tests for Simhash and CrossPolytope`

---

## Step 3: Simhash via `RandomDotProducts`

**Goal:** Simhash computes its `K·L` projections with one `RandomDotProducts`
over `max(dimensions, K·L)` floats, rounded up to a power of two (16384 for
EMST's K = 24, L = 512): three rounds of random sign flips and FHTs, exactly
as `LatticeLSH` does. Today it computes 12288 int16 dot products against a
2.75 MB matrix that does not fit in L2. Expected: about 2× faster hashing.

**Files:** `include/panna/linalg.hpp`, `include/panna/lsh/simhash.hpp`,
`test/include/lsh/hash_test.hpp`.

### 3a. `RandomDotProducts`: add `size()` and `rotate_unscaled()`

Split `compute` into the rotation (`rotate_unscaled`) and the final scaling,
so that Simhash can skip the scaling (signs are invariant to it). The
behaviour of `compute` must not change. Apply this diff to
`include/panna/linalg.hpp` (class `RandomDotProducts`, around line 505):

```diff
@@ -505,10 +505,20 @@ namespace panna {
             return scratch;
         }
 
-        void compute( std::vector<float>& in_out, float additional_scaling = 1.0 ) const {
+        //! The number of outputs, i.e. the requested number of products
+        //! rounded up to a power of two. Buffers passed to `compute` and
+        //! `rotate_unscaled` must have exactly this size.
+        size_t size() const {
+            return num_products;
+        }
+
+        //! The three rounds of (random +-1 diagonal, fast Hadamard transform)
+        //! of `compute`, without its final normalization: every output is
+        //! `num_products / additional_scaling` times the corresponding output
+        //! of `compute`. Use it when only the signs or the argmax of the
+        //! outputs matter, since they are invariant to a positive scaling.
+        void rotate_unscaled( std::vector<float>& in_out ) const {
             expect( in_out.size() == num_products );
-            // float norm_factor = std::sqrt(num_products) / (num_products * std::sqrt(static_cast<double>(num_products)));
-            float norm_factor = additional_scaling / static_cast<float>(num_products);
             const size_t padded_size = 1u << log_num_products;
             for ( uint8_t diagonal = 0; diagonal < 3; diagonal++ ) {
                 // Multiply by a diagonal +-1 matrix.
@@ -533,6 +543,11 @@ namespace panna {
                 // Apply the fast hadamard transform
                 fht( in_out.data(), log_num_products );
             }
+        }
+
+        void compute( std::vector<float>& in_out, float additional_scaling = 1.0 ) const {
+            rotate_unscaled( in_out );
+            const float norm_factor = additional_scaling / static_cast<float>( num_products );
 #ifdef __AVX2__
             float* values = in_out.data();
             __m256 scale = _mm256_set1_ps( norm_factor );
```

The rest of `compute` (the AVX2 / scalar scaling loops) is unchanged. After
the edit, the class has, in order: `allocate_scratch()`, `size()`,
`rotate_unscaled()` (the old loop over the three diagonals) and `compute()`
(a call to `rotate_unscaled`, then the scaling loops).

### 3b. Replace `include/panna/lsh/simhash.hpp`

Replace the whole file with:

```cpp
#pragma once

#include <algorithm>
#include <sstream>
#include <vector>

#include "panna/linalg.hpp"
#include "panna/lsh/values.hpp"

namespace panna {

    template <uint8_t K, typename Dataset, typename Distance>
    class SimhashBuilder;

    //! SimHash (random hyperplanes) for the angular / cosine distance.
    //!
    //! The `repetitions * K` random projections of a point are computed all at
    //! once by a `RandomDotProducts` (three rounds of random sign flips and
    //! fast Hadamard transforms over a buffer of `max(dimensions, K *
    //! repetitions)` floats, rounded up to a power of two), as `LatticeLSH`
    //! does. Projection `rep * K + concat` gives bit `concat` of repetition
    //! `rep`.
    template <uint8_t K, typename Dataset, typename Distance>
    class Simhash {
    public:
        //! The datatype of the output
        using Value = BitwiseLshValue<K>;
        using Builder = SimhashBuilder<K, Dataset, Distance>;

    private:
        size_t dimensions = 0;
        size_t repetitions = 0;
        RandomDotProducts random_dots;

    public:
        Simhash() {
        }

        Simhash( size_t dimensions, size_t repetitions ):
            dimensions( dimensions ),
            repetitions( repetitions ),
            random_dots( std::max( dimensions, repetitions * K ) ) {
        }

        friend bool operator==( const Simhash<K, Dataset, Distance>& a,
                                const Simhash<K, Dataset, Distance>& b ) {
            return a.dimensions == b.dimensions && a.repetitions == b.repetitions &&
                   a.random_dots == b.random_dots;
        }

        static constexpr size_t get_concatenations() {
            return K;
        }

        size_t get_repetitions() const {
            return repetitions;
        }

        //! Thread safe: the scratch buffer is `thread_local`, so `hash` can be
        //! called concurrently from any number of threads, OpenMP or not.
        void hash( typename Dataset::PointHandle point, std::vector<Value>& output ) const {
            thread_local std::vector<float> scratch;
            // Reset the whole buffer: it may hold the projections of the
            // previous point, or have been used by a hasher of another size.
            scratch.assign( random_dots.size(), 0.0f );
            point.into_vec( scratch );
            random_dots.rotate_unscaled( scratch );

            output.resize( repetitions );
            const float* projections = scratch.data();
            for ( size_t rep = 0; rep < repetitions; rep++ ) {
                uint32_t bits = 0;
                // concatenation 0 ends up as the most significant of the K bits
                for ( size_t concat = 0; concat < K; concat++ ) {
                    bits = ( bits << 1 ) | ( projections[rep * K + concat] > 0.0f );
                }
                output[rep] = Value::make( bits );
            }
        }

        float collision_probability( float distance ) const {
            float angle = Distance::to_angle( distance );
            return 1.0 - angle / M_PI;
        }
    };

    template <uint8_t K, typename Dataset, typename Distance>
    class SimhashBuilder {
        size_t dimensions = 0;

    public:
        using Output = Simhash<K, Dataset, Distance>;

        //! The hash function does not rescale with a target distance:
        //! all the `fit` methods are no-ops.
        static constexpr bool fits_to_distance = false;

        SimhashBuilder() {}

        SimhashBuilder( size_t dimensions ): dimensions( dimensions ) {
        }

        void reset() {
        }

        void fit( Dataset&, size_t, float ) {
        }

        void fit( const Dataset&, float, size_t, float ) {
        }

        Output build( size_t repetitions ) const {
            return Simhash<K, Dataset, Distance>( dimensions, repetitions );
        }

        std::string describe() const {
            return "Simhash";
        }
    };
} // namespace panna
```

Notes on why it is written this way:
- **`hash` stays `const`.** `test/include/pairforest_test.hpp:509` calls
  `forest.get_hasher().hash(...)` through a const reference, and
  `python/wrapper.cpp` treats const hashers as thread safe (the
  `ConstHasher` concept). The scratch buffer is therefore a function-local
  `thread_local`. Do not use per-thread vectors indexed by
  `omp_get_thread_num()`: those break when more threads hash than existed at
  construction.
- `scratch.assign( size, 0 )` resets the buffer on every call, because the
  same thread may use hashers of different sizes.
- **Bit order is part of the format.** Concatenation 0 must be the most
  significant of the K bits (`BitwiseLshValue::prefix_eq` relies on it),
  hence `bits = ( bits << 1 ) | ...` with `concat` going up.
- `collision_probability` is unchanged (`1 − θ/π`).
- The `random_vectors` member and the `Dataset` copy are gone. Nothing
  outside this file used them.

### 3c. Make a flaky test's tolerance statistical

`TEST_CASE( "Failure probability independent" )` (in
`test/include/lsh/hash_test.hpp`, function `test_failure_probability`)
estimates the failure probability of each of 24 × 128 (k, repetition) cells
from 1000 samples. It requires `actual < expected + 0.03`, but the standard
error is up to 0.016, so the test fails at random. Measured over 8 seeds,
the **old** Simhash failed at seed 2 (excess 0.037), and the new one at
seeds 1234 (the default) and 4 (excess 0.0304 and 0.0305). In
`test_failure_probability`, replace:

```cpp
                    // We check that the actual failure probability is lower than the expected one
                    REQUIRE( actual < exp + 0.03 );
```

with:

```cpp
                    // We check that the actual failure probability is lower than the expected one,
                    // up to 4 standard errors of an estimate from `samples_per_experiment` samples
                    const float tolerance =
                        4.0f * std::sqrt( exp * ( 1.0f - exp ) / samples_per_experiment ) + 0.005f;
                    REQUIRE( actual < exp + tolerance );
```

### 3d. Build, test, benchmark

```sh
nix develop -c cmake --build build/cpp -j --target tests bench-lsh-build
nix develop -c build/cpp/tests "[lsh-quality]"
nix develop -c build/cpp/tests "*Simhash*,*CrossPolytope*,*ensor*,Failure probability*"
nix develop -c build/cpp/tests "[pairforest]"
nix develop -c build/cpp/tests "*trie*"
nix develop -c build/cpp/tests "[pairemst]"
nix develop -c build/cpp/tests                       # only the pre-existing failures
nix develop -c build/cpp/bench-lsh-build datasets/glove-100-angular.hdf5 100000 simhash parallel
nix develop -c .venv/bin/pip install .
nix develop -c .venv/bin/python -m pytest -q python/tests/test_lsh.py
```

**Expected:**
- the Simhash quality test passes (CP still "may fail");
- every listed subset passes;
- the benchmark gives about `simhash K=24 build 0.5 ms | hash ~1340 ms`
  (against about 3340 ms in step 1), and 1-thread pinned about 35 µs/point
  (against about 72);
- 127 Python tests pass.

**Commit:** `compute Simhash projections with RandomDotProducts`

---

## Step 4: CrossPolytope, lean implementation (3 rounds, same distribution)

**Goal:** the same hash family (3 independent pseudo-random rotation rounds
per symbol, then the closest signed axis), implemented without the current
overheads. Per point, the current code:
- converts the point from int16 2048 times instead of once;
- reads 2048 separate `RandomDotProducts`, whose ±1 signs are stored as
  floats (3 MB in total), so every point streams them from L3/DRAM. This is
  why it scales only 2× with 16 threads;
- multiplies by a normalization factor that the argmax does not need;
- finds the argmax with a branchy scalar loop full of `.at()`;
- has the 100-of-128-axes bug described in step 2.

The new code stores the signs as **bits** (one table of
`K·L·3·padded/8` bytes, 96 KB for EMST, L2-resident). It converts the point
once, applies the signs with an XOR on the float sign bit (AVX2), and finds
the argmax with AVX2 over **all** `padded` coordinates. Expected: about 2.5×
faster hashing in parallel, and about 1.7× on a single thread.

**Files:** `include/panna/lsh/crosspolytope.hpp`,
`test/include/lsh/quality_test.hpp`, `python/wrapper.cpp` (comments and
docstrings only).

### 4a. Replace `include/panna/lsh/crosspolytope.hpp`

Replace the whole file with the content below.
`CrossPolytopeCollisionEstimates` is **unchanged** from the current file
(step 5 rewrites it), and so is `CrossPolytopeBuilder` (default
`estimation_repetitions = 1024`).

```cpp
#pragma once
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <omp.h>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "panna/linalg.hpp"
#include "panna/lsh/values.hpp"
#include "panna/rand.hpp"

#if defined( __AVX2__ )
    #include <immintrin.h>
#endif

namespace panna {

    struct CrossPolytopeCollisionEstimates {
        std::vector<float> probabilities;
        float eps;

        CrossPolytopeCollisionEstimates() {
        }

        friend bool operator==( const CrossPolytopeCollisionEstimates& a,
                                const CrossPolytopeCollisionEstimates& b ) {
            return a.probabilities == b.probabilities && a.eps == b.eps;
        }

        CrossPolytopeCollisionEstimates( unsigned int dimensions,
                                         unsigned int num_repetitions,
                                         float eps ):
            eps( eps ) {
            // adapted from
            // https://github.com/puffinn/puffinn/blob/master/include/puffinn/hash/crosspolytope.hpp

            auto log_dimensions = ceil_log( dimensions );
            probabilities = std::vector<float>();
            size_t num_probabilities = 2 / eps;
            probabilities.resize( num_probabilities );

            // foreach [alpha, alpha+eps) segment
#pragma omp parallel for
            for ( size_t i = 0; i < num_probabilities; i++ ) {
                float alpha = -1 + eps * i;
                std::normal_distribution<double> standard_normal( 0, 1 );
                std::mt19937_64 rng;
                rng.seed( 1234 + i );

                size_t collisions = 0;

                for ( size_t z = 0; z < num_repetitions; z++ ) {
                    // length = dimensions
                    // x = (1, 0, ..., 0)
                    // y = (alpha, (1-alpha^2)^(1/2), ..., 0)
                    // The dot product between x and y is alpha, and both are
                    // unit norm.

                    // The hash value so far.
                    uint32_t hash_x = 0;
                    uint32_t hash_y = 0;
                    // Absolute value of highest value seen.
                    double v_x = 0;
                    double v_y = 0;

                    // Compute a random rotation of x and y using the matrix z
                    // [ [ z_1_0, z_2_0 ],
                    //   [ z_1_1, z_2_1 ],
                    //   [ z_1_j, z_2_j ] ]
                    for ( uint32_t j = 0; j < dimensions; j++ ) {
                        double z_1 = standard_normal( rng );
                        double z_2 = standard_normal( rng );
                        // calculate z*x[j] and find the index with the highest
                        // value
                        if ( abs( z_1 ) > v_x ) {
                            v_x = abs( z_1 );
                            hash_x = j;
                            if ( z_1 < 0 ) {
                                hash_x |= ( 1 << log_dimensions );
                            }
                        }
                        // do the same for z*y[j]
                        double h_y = alpha * z_1 + pow( 1 - pow( alpha, 2 ), 0.5 ) * z_2;
                        if ( abs( h_y ) > v_y ) {
                            v_y = abs( h_y );
                            hash_y = j;
                            if ( h_y < 0 ) {
                                hash_y |= ( 1 << log_dimensions );
                            }
                        }
                    }
                    collisions += hash_x == hash_y;
                }
                float prob;
                if ( num_repetitions != 0 ) {
                    prob = static_cast<float>( collisions ) / num_repetitions;
                } else {
                    prob = 1.0;
                }
                assert( prob <= 1.0 );
                probabilities.at(i) = prob;
            }
        }

        float get_collision_probability( float dotp ) const {
            // Callers may probe distances outside the valid domain of the
            // metric (e.g. the binary search in distance_at_failure_probability),
            // so clamp to the boundary segments instead of indexing out of range.
            const float raw_idx = std::floor( ( 1.0 + dotp ) / eps );
            const size_t max_idx = probabilities.size() - 1;
            if ( raw_idx < 0.0 ) {
                return probabilities.at( 0 );
            }
            const size_t idx = std::min( static_cast<size_t>( raw_idx ), max_idx );
            return probabilities.at( idx );
        }
    };

    namespace crosspolytope_detail {

        //! Flips the sign of `v[i]` for every set bit `i` of `bits` (bit
        //! `i % 8` of byte `i / 8`), for `i < n`. `n` is a multiple of 8.
        inline void flip_signs( float* v, const uint8_t* bits, size_t n ) {
#ifdef __AVX2__
            // lane j of a group of 8 corresponds to bit j of the byte
            const __m256i select = _mm256_setr_epi32( 1, 2, 4, 8, 16, 32, 64, 128 );
            const __m256i sign_bit = _mm256_set1_epi32( static_cast<int>( 0x80000000u ) );
            for ( size_t i = 0; i < n; i += 8 ) {
                const __m256i byte = _mm256_set1_epi32( bits[i / 8] );
                const __m256i set =
                    _mm256_cmpeq_epi32( _mm256_and_si256( byte, select ), select );
                const __m256 mask = _mm256_castsi256_ps( _mm256_and_si256( set, sign_bit ) );
                _mm256_storeu_ps( v + i, _mm256_xor_ps( _mm256_loadu_ps( v + i ), mask ) );
            }
#else
            for ( size_t i = 0; i < n; i++ ) {
                if ( ( bits[i / 8] >> ( i % 8 ) ) & 1 ) {
                    v[i] = -v[i];
                }
            }
#endif
        }

        //! The cross-polytope code of `v` (of length `n`, a multiple of 8):
        //! `i` if `v[i]` is the first entry of largest magnitude and it is
        //! non-negative, `i + n` if it is negative.
        inline int16_t closest_axis( const float* v, size_t n ) {
#ifdef __AVX2__
            const __m256 abs_mask = _mm256_castsi256_ps( _mm256_set1_epi32( 0x7fffffff ) );
            __m256 max8 = _mm256_setzero_ps();
            for ( size_t i = 0; i < n; i += 8 ) {
                max8 = _mm256_max_ps( max8, _mm256_and_ps( _mm256_loadu_ps( v + i ), abs_mask ) );
            }
            __m128 max4 = _mm_max_ps( _mm256_castps256_ps128( max8 ), _mm256_extractf128_ps( max8, 1 ) );
            max4 = _mm_max_ps( max4, _mm_movehl_ps( max4, max4 ) );
            max4 = _mm_max_ss( max4, _mm_shuffle_ps( max4, max4, 1 ) );
            const __m256 best = _mm256_set1_ps( _mm_cvtss_f32( max4 ) );
            for ( size_t i = 0; i < n; i += 8 ) {
                const __m256 a = _mm256_and_ps( _mm256_loadu_ps( v + i ), abs_mask );
                const int hits = _mm256_movemask_ps( _mm256_cmp_ps( a, best, _CMP_EQ_OQ ) );
                if ( hits != 0 ) {
                    const size_t j = i + __builtin_ctz( hits );
                    return static_cast<int16_t>( v[j] >= 0.0f ? j : j + n );
                }
            }
            return 0; // only reachable with NaNs
#else
            size_t code = 0;
            float best = -1.0f;
            for ( size_t i = 0; i < n; i++ ) {
                const float a = std::fabs( v[i] );
                if ( a > best ) {
                    best = a;
                    code = v[i] >= 0.0f ? i : i + n;
                }
            }
            return static_cast<int16_t>( code );
#endif
        }

    } // namespace crosspolytope_detail

    template <uint8_t K, typename Dataset, typename Distance>
    class CrossPolytopeBuilder;

    //! Cross-polytope LSH. Each of the `K * repetitions` symbols applies its
    //! own pseudo-random rotation to the point -- `ROUNDS` rounds of (random
    //! +-1 diagonal, fast Hadamard transform) over the point zero-padded to
    //! `padded` dimensions -- and outputs the closest signed axis.
    template <uint8_t K, typename Dataset, typename Distance>
    class CrossPolytope {
    public:
        //! The datatype of the output
        using Value = ShortLshValue<K>;
        using Builder = CrossPolytopeBuilder<K, Dataset, Distance>;
        //! Number of (sign flip, FHT) rounds of every pseudo-random rotation.
        static constexpr size_t ROUNDS = 3;
        //! Codes are in `[0, 2 * padded)` and must fit an `int16_t`.
        static constexpr size_t MAX_PADDED_DIMENSIONS = 16384;

    private:
        size_t repetitions = 0;
        size_t dimensions = 0;
        //! log2 of `padded`
        unsigned int log_padded = 0;
        //! `dimensions` rounded up to a power of two, and to at least 8 (the
        //! width of an AVX2 register of floats)
        size_t padded = 0;
        //! The random diagonals, one bit per entry: the `ROUNDS` diagonals of
        //! symbol `rep * K + concat` are the `ROUNDS * padded / 8` bytes
        //! starting at byte `(rep * K + concat) * ROUNDS * padded / 8`.
        std::vector<uint8_t> sign_bits;
        CrossPolytopeCollisionEstimates estimates;

        static unsigned int checked_log_padded( size_t dimensions ) {
            const unsigned int log = std::max( 3u, ceil_log( dimensions ) );
            if ( ( size_t( 1 ) << log ) > MAX_PADDED_DIMENSIONS ) {
                throw std::invalid_argument( "CrossPolytope: at most " +
                                             std::to_string( MAX_PADDED_DIMENSIONS ) +
                                             " dimensions are supported" );
            }
            return log;
        }

    public:
        static constexpr size_t get_concatenations() {
            return K;
        }

        CrossPolytope() {
        }

        CrossPolytope( size_t dimensions,
                       size_t repetitions,
                       size_t estimation_repetitions = 8192,
                       float estimation_eps = 5e-3 ):
            repetitions( repetitions ),
            dimensions( dimensions ),
            log_padded( checked_log_padded( dimensions ) ),
            padded( size_t( 1 ) << log_padded ),
            estimates( padded, estimation_repetitions, estimation_eps ) {
            sign_bits.resize( K * repetitions * ROUNDS * ( padded / 8 ) );
            auto& rng = get_global_rng();
            for ( uint8_t& byte : sign_bits ) {
                byte = static_cast<uint8_t>( rng() & 0xff );
            }
        }

        friend bool operator==( const CrossPolytope<K, Dataset, Distance>& a,
                                const CrossPolytope<K, Dataset, Distance>& b ) {
            return a.repetitions == b.repetitions && a.dimensions == b.dimensions &&
                   a.sign_bits == b.sign_bits && a.estimates == b.estimates;
        }

        size_t get_repetitions() const {
            return repetitions;
        }

        //! Requires `Dataset::PointHandle::into_vec`, which copies the point
        //! into the first `dimensions` entries of the given vector.
        //!
        //! Thread safe: the scratch buffers are `thread_local`.
        void hash( typename Dataset::PointHandle point, std::vector<Value>& output ) const {
            thread_local std::vector<float> base;
            thread_local std::vector<float> work;
            base.assign( padded, 0.0f );
            work.resize( padded );
            point.into_vec( base );

            output.resize( repetitions );
            const size_t bytes_per_round = padded / 8;
            const uint8_t* bits = sign_bits.data();
            for ( size_t rep = 0; rep < repetitions; rep++ ) {
                Value cur;
                for ( size_t concat = 0; concat < K; concat++ ) {
                    std::copy( base.begin(), base.end(), work.begin() );
                    for ( size_t round = 0; round < ROUNDS; round++ ) {
                        crosspolytope_detail::flip_signs( work.data(), bits, padded );
                        bits += bytes_per_round;
                        fht( work.data(), log_padded );
                    }
                    // The FHT is unnormalized, but the closest axis does not
                    // depend on a positive scaling.
                    cur.set( concat, crosspolytope_detail::closest_axis( work.data(), padded ) );
                }
                output[rep] = cur;
            }
        }

        float collision_probability( float distance ) const {
            float dotp = Distance::to_dot_product( distance );
            return estimates.get_collision_probability( dotp );
        }
    };

    template <uint8_t K, typename Dataset, typename Distance>
    class CrossPolytopeBuilder {
        size_t dimensions = 0;
        size_t estimation_repetitions = 1024;
        float estimation_eps = 5e-3;

    public:
        using Output = CrossPolytope<K, Dataset, Distance>;

        //! The hash function does not rescale with a target distance:
        //! all the `fit` methods are no-ops.
        static constexpr bool fits_to_distance = false;

        CrossPolytopeBuilder() {
        }

        CrossPolytopeBuilder( size_t dimensions,
                              size_t estimation_repetitions = 1024,
                              float estimation_eps = 5e-3 ):
            dimensions( dimensions ),
            estimation_repetitions( estimation_repetitions ),
            estimation_eps( estimation_eps ) {
        }

        void reset() {
        }

        void fit( Dataset&, size_t, float ) {
        }

        void fit( const Dataset&, float, size_t, float ) {
        }

        Output build( size_t repetitions ) const {
            return CrossPolytope<K, Dataset, Distance>(
                dimensions, repetitions, estimation_repetitions, estimation_eps );
        }

        std::string describe() const {
            std::stringstream sstream;
            sstream << "CrossPolytope";
            return sstream.str();
        }
    };

} // namespace panna
```

Notes:
- **`padded` = `max(8, next power of two ≥ dimensions)`.** The minimum of 8
  is the width of an AVX2 register, so the vector loops need no tail
  handling. The estimates are now built for `padded` axes, which is also the
  number of axes the argmax scans. For `dimensions >= 8` this is the same
  `1 << ceil_log(dimensions)` as before. The codes are in `[0, 2·padded)`,
  instead of `[0, 2·dimensions)` with the old encoding.
- **Sign-bit layout:** symbol `rep·K + concat`, round `r` uses the
  `padded/8` bytes starting at byte `((rep·K + concat)·3 + r)·padded/8`.
  Within a byte, bit `j` flips coordinate `8·i + j` (that is what
  `_mm256_setr_epi32(1, 2, 4, …, 128)` in `flip_signs` encodes). The
  scalar fallback uses the same layout, and the `hash` loop walks the table
  sequentially.
- **The argmax tie-breaking is unchanged:** the first coordinate of maximum
  magnitude wins, and a zero vector gives code 0.
- `hash` is now **`const`** with `thread_local` scratch, like Simhash.
  `python/wrapper.cpp`'s `hash_rows` will therefore treat CP as a
  `ConstHasher` (no mutex), which is correct now. The old per-thread vectors
  (`tl_rotated_vectors`), `random_dots`, `hash_single` and
  `encode_closest_axis` are gone.
- Hash codes for a given seed change, because the RNG is consumed
  differently. No test depends on specific values.
- `include/panna/pairforest.hpp` (around line 688) has a comment saying that
  `CrossPolytope::hash` is non-const. It is now stale, but that file is off
  limits: leave it.

### 4b. Remove `[!mayfail]` from the CP quality test

In `test/include/lsh/quality_test.hpp`, change

```cpp
    TEST_CASE( "CrossPolytope collision quality", "[lsh-quality][!mayfail]" ) {
```

to

```cpp
    TEST_CASE( "CrossPolytope collision quality", "[lsh-quality]" ) {
```

### 4c. Update `python/wrapper.cpp` comments and docstrings

Only comments and strings change. Replace, exactly:

1. Around line 305:

```cpp
/// Hashers whose `hash` can be called on a `const` object. The others
/// (`CrossPolytope`, `LatticeLSH`) use per-thread scratch space.
```

with

```cpp
/// Hashers whose `hash` can be called on a `const` object. The others
/// (`LatticeLSH`) use per-thread scratch space.
```

2. Around line 317:

```cpp
/// `max_threads` must be at most the number of threads the hasher was
/// constructed for: `CrossPolytope` and `LatticeLSH` index their scratch space,
```

with

```cpp
/// `max_threads` must be at most the number of threads the hasher was
/// constructed for: `LatticeLSH` indexes its scratch space,
```

3. Around line 613:

```cpp
    /// The codes, in `[0, 2 * dimensions)`, are stored as `int16`.
```

with

```cpp
    /// The codes, in `[0, 2 * padded)` where `padded` is `dimensions` rounded
    /// up to a power of two (at least 8), are stored as `int16`.
```

4. In the `CrossPolytope` class docstring (around line 1899), replace the line

```cpp
        "an integer in [0, 2 * dimensions). Input points are normalized to unit\n"
```

   with the two lines

```cpp
        "an integer in [0, 2 * p), where p is `dimensions` rounded up to a power\n"
        "of two (at least 8). Input points are normalized to unit\n"
```

5. In the `hash` docstring (around line 1939), replace the line

```cpp
               "  holding the code, in [0, 2 * dimensions), of each concatenation. If\n"
```

   with

```cpp
               "  holding the code, in [0, 2 * p), of each concatenation. If\n"
```

(The `.pyi` stub is regenerated from these strings at build time. Do not edit
it by hand.)

### 4d. Build, test, benchmark

Same commands as in 3d, but with `crosspolytope` in the benchmark:

```sh
nix develop -c cmake --build build/cpp -j --target tests bench-lsh-build
nix develop -c build/cpp/tests "[lsh-quality]"
nix develop -c build/cpp/tests "*Simhash*,*CrossPolytope*,*ensor*,Failure probability*"
nix develop -c build/cpp/tests "[pairforest]"
nix develop -c build/cpp/tests "[pairemst]"
nix develop -c build/cpp/tests                       # only the pre-existing failures
nix develop -c build/cpp/bench-lsh-build datasets/glove-100-angular.hdf5 100000 crosspolytope parallel
for t in 1 2 3; do OMP_NUM_THREADS=1 nix develop -c taskset -c 2 build/cpp/bench-lsh-build datasets/glove-100-angular.hdf5 6000 crosspolytope single; done
nix develop -c .venv/bin/pip install .
nix develop -c .venv/bin/python -m pytest -q python/tests/test_lsh.py
```

**Expected:**
- **both** quality tests pass. The CP symbol rates are now within about
  0.002 of the reference;
- the benchmark gives a CP hash time of about 14.3 s (against about 36 s),
  and 1-thread pinned about 440 µs/point (against about 754). The `build`
  time is still about 2.4 s; step 5 fixes it;
- 127 Python tests pass. This includes `test_scratch_built_with_one_thread`
  (hashing with more threads than at construction) and
  `test_cosine_scale_invariance`, and the CP-specific error tests with
  `dimensions = 4` (padded to 8).

**Commit:** `rewrite CrossPolytope hashing with bit-packed signs and a vectorized argmax`

---

## Step 5: cheaper CrossPolytope collision estimates

**Goal:** `CrossPolytopeCollisionEstimates` runs a Monte Carlo simulation
every time a hasher is built. EMST builds one, and it counts towards
`index_ms`. Today it draws `2 · padded` normals for **each** of the 400 table
entries and each of the `estimation_repetitions` samples (about 105 M
normals with the builder's 1024 samples), which takes about 2.4 s
single-threaded and about 0.75 s on 16 threads. Instead, each sample draws
its two Gaussian columns **once** and evaluates every table entry with them
("common random numbers"). That is 400× fewer normals, and it also makes the
table smooth in the dot product. The construction becomes cheap enough to
raise the builder's default from 1024 to 8192 samples. 8192 is already the
`CrossPolytope` constructor's default, so the two now agree, and the
estimates become more accurate: the standard error per entry drops from
about 0.016 to about 0.006.

**Files:** `include/panna/lsh/crosspolytope.hpp`.

1. Replace the whole `struct CrossPolytopeCollisionEstimates { ... };`
   (from `struct CrossPolytopeCollisionEstimates {` to the closing `};`
   after `get_collision_probability`) with this block, doc comment
   included:

```cpp
    //! Monte Carlo estimates of the probability that two unit vectors with a
    //! given dot product get the same cross-polytope hash, tabulated at the
    //! dot products `-1 + i * eps`.
    //!
    //! The hash is modelled with a `dimensions x dimensions` matrix of i.i.d.
    //! standard normals. Each Monte Carlo sample draws the first two columns
    //! `z1`, `z2` of such a matrix (from its own seed, `1234 + sample`) and
    //! evaluates the hashes of `x = e1` and of `y = alpha e1 + sqrt(1 -
    //! alpha^2) e2` for *every* tabulated `alpha`. Reusing the sample across
    //! the table (common random numbers) costs `2 * dimensions` normals per
    //! sample rather than per sample and table entry, and makes the table
    //! smooth in `alpha`.
    struct CrossPolytopeCollisionEstimates {
        std::vector<float> probabilities;
        float eps;

        CrossPolytopeCollisionEstimates() {
        }

        friend bool operator==( const CrossPolytopeCollisionEstimates& a,
                                const CrossPolytopeCollisionEstimates& b ) {
            return a.probabilities == b.probabilities && a.eps == b.eps;
        }

        CrossPolytopeCollisionEstimates( unsigned int dimensions,
                                         unsigned int num_repetitions,
                                         float eps ):
            eps( eps ) {
            const size_t num_probabilities = 2 / eps;
            // With no samples every entry is 1, as before.
            probabilities.assign( num_probabilities, 1.0f );
            if ( num_repetitions == 0 ) {
                return;
            }

            // The code of the closest signed axis to `v`: `j` if `v[j]` is the
            // entry of largest magnitude (the first one, in case of ties) and
            // it is non-negative, `j + dimensions` if it is negative.
            auto closest_axis = [dimensions]( const float* v ) {
                uint32_t code = 0;
                float best = -1.0f;
                for ( uint32_t j = 0; j < dimensions; j++ ) {
                    const float a = std::fabs( v[j] );
                    if ( a > best ) {
                        best = a;
                        code = v[j] < 0.0f ? j + dimensions : j;
                    }
                }
                return code;
            };

            std::vector<size_t> collisions( num_probabilities, 0 );
#pragma omp parallel
            {
                std::vector<size_t> local_collisions( num_probabilities, 0 );
                std::vector<float> z1( dimensions ), z2( dimensions ), y( dimensions );
#pragma omp for schedule( static )
                for ( size_t sample = 0; sample < num_repetitions; sample++ ) {
                    std::mt19937_64 rng( 1234 + sample );
                    std::normal_distribution<float> standard_normal( 0.0f, 1.0f );
                    for ( uint32_t j = 0; j < dimensions; j++ ) {
                        z1[j] = standard_normal( rng );
                        z2[j] = standard_normal( rng );
                    }
                    // the rotation of x = e1 is z1 itself
                    const uint32_t hash_x = closest_axis( z1.data() );
                    for ( size_t i = 0; i < num_probabilities; i++ ) {
                        const float alpha = -1.0f + eps * i;
                        const float beta = std::sqrt( std::max( 0.0f, 1.0f - alpha * alpha ) );
                        for ( uint32_t j = 0; j < dimensions; j++ ) {
                            y[j] = alpha * z1[j] + beta * z2[j];
                        }
                        local_collisions[i] += closest_axis( y.data() ) == hash_x;
                    }
                }
#pragma omp critical
                for ( size_t i = 0; i < num_probabilities; i++ ) {
                    collisions[i] += local_collisions[i];
                }
            }
            for ( size_t i = 0; i < num_probabilities; i++ ) {
                probabilities[i] =
                    static_cast<float>( collisions[i] ) / static_cast<float>( num_repetitions );
            }
        }

        float get_collision_probability( float dotp ) const {
            // Callers may probe distances outside the valid domain of the
            // metric (e.g. the binary search in distance_at_failure_probability),
            // so clamp to the boundary segments instead of indexing out of range.
            const float raw_idx = std::floor( ( 1.0 + dotp ) / eps );
            const size_t max_idx = probabilities.size() - 1;
            if ( raw_idx < 0.0 ) {
                return probabilities.at( 0 );
            }
            const size_t idx = std::min( static_cast<size_t>( raw_idx ), max_idx );
            return probabilities.at( idx );
        }
    };
```

   Notes:
   - Each sample has its own seed (`1234 + sample`), so the table does not
     depend on the number of threads.
   - With `num_repetitions == 0` every entry is 1, as before.
   - `get_collision_probability` is unchanged.
   - The constructor's signature is unchanged. Python's
     `pp.CrossPolytope(..., estimation_repetitions=..., estimation_eps=...)`
     keeps working, and so do its tests, which pass explicit values.

2. In `class CrossPolytopeBuilder`, change both defaults from 1024 to 8192:

```cpp
        size_t estimation_repetitions = 8192;
```

   and

```cpp
        CrossPolytopeBuilder( size_t dimensions,
                              size_t estimation_repetitions = 8192,
                              float estimation_eps = 5e-3 ):
```

   Do **not** change the Python binding's default (`estimation_repetitions =
   1024` in `python/wrapper.cpp`): it is a public API default and is out of
   scope.

3. Build, test, benchmark: the same commands as in 4d.

**Expected:** all as in step 4, and the benchmark shows a CP `build` time of
about 150 ms (against about 2.4 s), with 8192 samples instead of 1024.

**Commit:** `estimate CrossPolytope collision probabilities with common random numbers`

---

## Step 6: end-to-end check, CHANGELOG, plan status

1. Measure end to end with the installed extension (it must be rebuilt
   after step 5: `nix develop -c .venv/bin/pip install .`). Save this
   script as `scripts/bench_lsh_index.py`:

```python
"""End-to-end `pair_forest_emst` on a 100k sample of glove-100-angular, per
LSH family: index construction time, discovery time, distance count."""
import sys

import numpy as np
import pypanna as panna

N, SEED, REPS = 100_000, 1234, 512
EPSILONS = [float(e) for e in sys.argv[1].split(",")] if len(sys.argv) > 1 else [0.4, 0.1]

_, data = panna.datasets.load("glove-100-angular", normalize=True)
rng = np.random.default_rng(SEED)
data = np.ascontiguousarray(data[rng.choice(data.shape[0], N, replace=False)], dtype=np.float32)

print(f"{'family':14s} {'eps':>5s} {'index_ms':>9s} {'disc_ms':>9s} {'distances':>12s} {'weight':>10s}")
for eps in EPSILONS:
    for fam in ["lattice", "crosspolytope", "simhash"]:
        weights, _, st = panna.pair_forest_emst(data, epsilon=eps, delta=0.1,
                                                repetitions=REPS, family=fam)
        print(f"{fam:14s} {eps:5.2f} {st['index_ms']:9d} {st['discovery_ms']:9d} "
              f"{st['distance_count']:12d} {float(np.sum(weights)):10.2f}", flush=True)
```

   Run it, keeping only the table (the library logs to stdout too):

```sh
nix develop -c .venv/bin/python scripts/bench_lsh_index.py 0.4,0.1 | grep -E "^(family|lattice|crosspolytope|simhash) "
```

   **Expected (approximately):**

```
lattice         0.40      4656      7868    569787480   88651.13
crosspolytope   0.40     15895      4722    146802385   88704.37
simhash         0.40      2445     16534    294832456   88679.34
lattice         0.10      4791     33526   5931628176   88277.37
crosspolytope   0.10     15877     18319   2280135354   88294.98
simhash         0.10      2427     37804   2965891942   88295.05
```

   The weights of the three families at the same eps must agree within
   0.1%. The CP and Simhash distance counts should be in the same range as
   before (CP ≈ 150 M at 0.4 and ≈ 2.3–2.8 G at 0.1; Simhash ≈ 290 M and
   ≈ 2.9 G). Lattice's discovery numbers vary from run to run (its code did
   not change).

2. Add to the top of `CHANGELOG.md`, right after the introductory
   paragraph and before `## [0.1.0] - 2026-10-06`:

```markdown
## [Unreleased]

### ⚡ Performance

- `Simhash` computes all its projections with one `RandomDotProducts`
  (randomized Hadamard transforms) instead of one dot product per bit:
  about 2.5× faster hashing.
- `CrossPolytope` stores its random signs as bits, converts each point
  once and finds the closest axis with AVX2: about 2.5× faster hashing.
- `CrossPolytope` collision probabilities are estimated with common random
  numbers across the table: hasher construction is about 15× faster, and
  `CrossPolytopeBuilder` now uses 8192 estimation samples by default
  (was 1024).

### 🐛 Bug fixes

- `CrossPolytope` only looked for the closest axis among the first
  `dimensions` coordinates, rather than all the coordinates of the padded
  rotation, so its collision rate exceeded its estimated collision
  probability (by about 0.008 at d = 100).

### ⚠️ Changes

- `Simhash` and `CrossPolytope` produce different hash codes for the same
  seed. `CrossPolytope` codes are now in `[0, 2p)`, with `p` the dimension
  rounded up to a power of two (at least 8).
- `Simhash::hash` and `CrossPolytope::hash` are `const` and thread safe.
```

3. In this plan, change `status: proposed` to `status: done`, add
   `completed: <date>`, and replace the "after" column of "Expected outcome"
   with your measured numbers (from steps 3–6).

**Commit:** `update CHANGELOG and plan after the CrossPolytope/Simhash speedups`

---

## Background: diagnosis of the baseline (for reference)

Baseline end to end (`pair_forest_emst`, 100k glove, L = 512):

| family | eps | index_ms | discovery_ms | distances |
|---|---|---|---|---|
| lattice | 0.4 | 4,894 | 8,652 | 610 M |
| crosspolytope | 0.4 | 40,626 | 4,884 | 154 M |
| simhash | 0.4 | 4,793 | 16,664 | 284 M |
| lattice | 0.2 | 4,785 | 15,170 | 1,503 M |
| crosspolytope | 0.2 | 41,613 | 8,647 | 514 M |
| simhash | 0.2 | 4,693 | 27,860 | 1,233 M |
| lattice | 0.1 | 4,755 | 45,533 | 9,058 M |
| crosspolytope | 0.1 | 40,176 | 21,450 | 2,797 M |
| simhash | 0.1 | 4,545 | 37,877 | 2,924 M |

- **LatticeLSH** needs only 8 projections per symbol. It gets all
  `K·L·8 = 16384` of them from one `RandomDotProducts` (three
  rounds of random ±1 diagonals and FHTs), and its sign tables (196 KB) fit
  in L2.
- **CrossPolytope** needs a full 128-dimensional rotation per symbol:
  2048 symbols × 3 rounds = 6144 FHT-128 per point. perf: FHT about 43%,
  float sign multiplies about 25% (3 MB of float signs, streamed per point),
  scalar argmax about 10%, repeated int16→float conversion about 6%. The
  collision-estimate Monte Carlo adds about 0.75 s per index build.
- **Simhash** computes 12288 int16 dot products against a 2.75 MB matrix
  that does not fit in L2.

## Rejected alternatives (measured; do not implement)

- **CP from a single transform over all symbols** (one `RandomDotProducts`
  of `K·L·128 = 262144` floats, with bit-packed signs, then an argmax per
  128-float block): 1.5–2.5× *slower* than step 4. It does
  `3·N·log N ≈ 14 M` adds per point against 5.5 M for the per-symbol FHTs,
  and its 1 MB per-thread buffers do not stay in cache.
- **CP with one transform per repetition** (`K·128 = 512` floats, then K
  argmaxes): 1.52 against 1.51 Mcycles/point for the per-symbol layout. The
  fewer FHT calls are offset by the longer transforms. No gain.
- **Lane-parallel FHT** (8 symbols at once, one per AVX lane): a naive
  version was only about 5% faster than step 4. A register-blocked kernel
  might do better, but it is not part of this plan.
- **Fewer rotation rounds or shared rounds:** out of scope by decision. The
  rounds stay at 3.

## Risks and notes

- `thread_local` scratch (Simhash: `16384` floats = 64 KB per thread for
  EMST's parameters; CP: 2 × `padded` floats) lives until the thread exits.
  This is negligible.
- `CrossPolytope::MAX_PADDED_DIMENSIONS = 16384` (codes must fit in
  `int16_t`). The constructor throws `std::invalid_argument` above it. The
  Python binding already limits `dimensions` to 16384.
- Benchmarks on the development laptop (hybrid P/E cores, turbo) vary by up
  to 2× from run to run. Compare best-of-3, and use the pinned single-thread
  mode for small differences.
