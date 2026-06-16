#pragma once

// =============================================================================
// Phase 0 verification harness for the parallel-code reorganization.
// See TODO.md "Parallel code reorganization (refactor plan)" → Phase 0.
//
// Purpose: lock in a behavioural baseline BEFORE the Phase 1–5 refactors so that
// every later phase is provably non-regressing. Two kinds of check live here:
//
//   * INVARIANT  — the actual >=1-delta guarantee. Must hold after EVERY phase:
//                  realized approximation ratio <= 1+epsilon, and the fraction
//                  of seeds that violate it stays <= delta.
//   * PIN        — bit-for-bit weight match against the exact tree at epsilon=0.
//                  Pins behaviour for the behaviour-preserving phases (1, 3, 4).
//                  Phase 2 changes *when* we stop, so the PIN is expected to be
//                  relaxed/regenerated when Phase 2 lands (see TODO note there).
//
// Run only this harness:   ./tests "[phase0]"
// Run the perf baseline:    ./tests "[baseline]"   (hidden by default via [.])
// =============================================================================

#include <catch2/catch_test_macros.hpp>

#include <chrono>
#include <cstdlib>
#include <vector>

#include "panna/data.hpp"
#include "panna/distance.hpp"
#include "panna/emst.hpp"
#include "panna/lsh/lattice.hpp"
#include "panna/rand.hpp"

namespace panna {
    namespace phase0 {

        // --- Shared configuration -------------------------------------------
        // Kept small so the default `./tests` run stays fast. The larger sweep
        // that actually exercises the failure-rate bound lives behind a hidden
        // tag (see TODO at the failure-rate check below).
        using Dataset = EuclideanPoints;
        using Distance = EuclideanDistance;
        using Hasher = LatticeLSH<4, Dataset, Distance>;

        struct Config {
            size_t dimensions = 8;
            size_t n = 200;
            size_t repetitions = 512;
            float delta = 0.01f;
            float epsilon = 0.2f;
            // Seeds to sweep. TODO(phase0): for a real failure-rate validation
            // this must be >> 1/delta; this stub uses a handful and therefore
            // requires *zero* violations (see check below).
            std::vector<uint64_t> seeds = { 1, 2, 3, 4, 5 };
            // Numerical slack on the ratio comparison (float accumulation).
            float ratio_tol = 1e-4f;
        };

        // Build a fresh random instance for a given seed. Re-seeds the global
        // RNG so each iteration is reproducible and independent.
        inline EMST<Dataset, Hasher, Distance> make_instance( const Config& cfg, uint64_t seed ) {
            seed_global_rng( seed );
            std::vector<std::vector<float>> data;
            data.reserve( cfg.n );
            for ( size_t i = 0; i < cfg.n; i++ ) {
                data.push_back( sample_random_normal_vector( cfg.dimensions ) );
            }
            return EMST<Dataset, Hasher, Distance>(
                cfg.dimensions, cfg.repetitions, data, cfg.delta, cfg.epsilon );
        }

        inline float tree_weight( const std::vector<Edge>& tree ) {
            float w = 0.0f;
            for ( const auto& e : tree ) {
                w += e.weight;
            }
            return w;
        }

    } // namespace phase0

    // -----------------------------------------------------------------------
    // INVARIANT — EMST path: realized ratio <= 1+epsilon across the seed sweep,
    // and the failure rate stays within delta.
    // -----------------------------------------------------------------------
    TEST_CASE( "EMST Phase 0 — approximation invariant (EMST path)", "[emst][phase0][invariant]" ) {
        phase0::Config cfg; // epsilon = 0.2

        size_t violations = 0;
        for ( uint64_t seed : cfg.seeds ) {
            auto emst = phase0::make_instance( cfg, seed );

            const float exact_w = emst.exact_tree().first;
            const float approx_w = emst.find_tree().first;

            REQUIRE( exact_w > 0.0f );
            const float ratio = approx_w / exact_w;

            // Approximate weight must never undershoot the true MST weight and
            // must stay within the (1+epsilon) upper bound.
            CHECK( approx_w >= exact_w - cfg.ratio_tol );
            if ( ratio > 1.0f + cfg.epsilon + cfg.ratio_tol ) {
                violations++;
            }
        }

        // TODO(phase0): with a small seed set the only defensible bound is zero
        // violations. Scale `seeds` up (behind a hidden [.][phase0-sweep] tag)
        // and switch this to `violations <= delta * seeds.size()` for a genuine
        // failure-rate check.
        CHECK( violations == 0 );
    }

    // -----------------------------------------------------------------------
    // INVARIANT — mutual-reachability path.
    // -----------------------------------------------------------------------
    TEST_CASE( "EMST Phase 0 — approximation invariant (MR path)", "[emst][phase0][invariant]" ) {
        phase0::Config cfg;
        const size_t num_neighbors = 5;

        size_t violations = 0;
        for ( uint64_t seed : cfg.seeds ) {
            auto emst = phase0::make_instance( cfg, seed );

            const float exact_w = emst.exact_mutual_reachability_distance_tree( num_neighbors ).first;
            const auto probabilistic = emst.find_tree_mutual_reachability_distance( num_neighbors );
            const float approx_w = phase0::tree_weight( probabilistic.first );

            REQUIRE( exact_w > 0.0f );
            const float ratio = approx_w / exact_w;

            CHECK( approx_w >= exact_w - cfg.ratio_tol );
            if ( ratio > 1.0f + cfg.epsilon + cfg.ratio_tol ) {
                violations++;
            }
        }
        CHECK( violations == 0 );
    }

    // -----------------------------------------------------------------------
    // PIN — at epsilon=0 the approximate weight must equal the exact weight
    // bit-for-bit. This is the non-regression guard for the behaviour-preserving
    // phases (1, 3, 4): collector owns the tree, read-only snapshots, persistent
    // pool — none of which may change the result.
    //
    // EXPECTED TO CHANGE AT PHASE 2: the decide-cadence change can legitimately
    // stop at a (slightly) different repetition. When Phase 2 lands, relax this
    // to the (1+epsilon) invariant above and/or regenerate a golden value.
    // -----------------------------------------------------------------------
    TEST_CASE( "EMST Phase 0 — behaviour pin (epsilon=0 exact match)", "[emst][phase0][pin]" ) {
        phase0::Config cfg;
        cfg.epsilon = 0.0f;
        cfg.delta = 0.001f;

        for ( uint64_t seed : cfg.seeds ) {
            auto emst = phase0::make_instance( cfg, seed );

            const float exact_w = emst.exact_tree().first;
            const float approx_w = emst.find_tree().first;

            REQUIRE( approx_w == exact_w );
        }
    }

    // -----------------------------------------------------------------------
    // BASELINE — perf/work counters. Hidden by default ([.]) because it is a
    // measurement, not a pass/fail assertion. Run explicitly with "[baseline]".
    //
    // TODO(phase0): emit these as CSV (or write to a file) and diff against a
    // committed baseline so Phases 1–5 can be compared on wall-clock,
    // distances-computed, and collisions. Also capture the PANNA_EMST_THREADS=1
    // single-threaded point to quantify the collector-vs-worker split (item #1).
    // -----------------------------------------------------------------------
    TEST_CASE( "EMST Phase 0 — perf baseline", "[.][emst][phase0][baseline]" ) {
        phase0::Config cfg;
        cfg.n = 2000; // a touch larger so timings are meaningful

        auto emst = phase0::make_instance( cfg, /*seed=*/1 );

        const auto t0 = std::chrono::steady_clock::now();
        const auto result = emst.find_tree();
        const auto t1 = std::chrono::steady_clock::now();
        const double elapsed_s = std::chrono::duration<double>( t1 - t0 ).count();

        WARN( "phase0-baseline"
              << " n=" << cfg.n
              << " weight=" << result.first
              << " elapsed_s=" << elapsed_s
              << " distances=" << emst.get_distance_count()
              << " collisions=" << emst.get_collisions_count() );

        // No assertion: this case exists to record a baseline, not to gate CI.
        SUCCEED();
    }

} // namespace panna
