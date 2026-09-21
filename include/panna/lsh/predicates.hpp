#pragma once

#include <cmath>
#include <cstddef>

#include "panna/expect.hpp"
#include "panna/lsh/tensoring.hpp"

namespace panna {

    //! Returns the probability for a pair at `distance` of not colliding, under
    //! the given hasher, in any repetition out of `max_repetitions`, where
    //! `repetitions` are done with `concatenations`, and the rest are done at
    //! `concatenations+1`.
    template <typename Hasher>
    static float failure_probability( Hasher& hasher,
                                      float distance,
                                      size_t concatenations,
                                      size_t repetitions,
                                      size_t max_repetitions ) {
        expect(repetitions <= max_repetitions);
        float collision_probability = hasher.collision_probability( distance );
        float p_cur =
            std::pow( 1 - std::pow( collision_probability, concatenations ), repetitions );
        float p_prev = 1.0;
        if ( concatenations + 1 <= hasher.get_concatenations() ) {
            p_prev = std::pow( 1 - std::pow( collision_probability, concatenations + 1 ),
                               max_repetitions - repetitions );
        }
        return p_cur * p_prev;
    }

    template <typename InnerHasher, typename Dataset>
    static float failure_probability( Tensoring<InnerHasher, Dataset>& hasher,
                                      float distance,
                                      size_t concatenations,
                                      size_t repetitions,
                                      size_t max_repetitions ) {
        expect(repetitions <= max_repetitions);
        auto cur_left_concatenations = ( concatenations + 1 ) / 2;
        auto cur_right_concatenations = concatenations - cur_left_concatenations;

        auto last_left_concatenations = ( concatenations + 2 ) / 2;
        auto last_right_concatenations = concatenations + 1 - last_left_concatenations;

        auto cur_repetitions = std::floor( std::sqrt( repetitions ) );
        auto last_repetitions = std::floor( std::sqrt( max_repetitions ) ) - cur_repetitions;

        auto left_prob =
            std::pow( hasher.collision_probability( distance ), cur_left_concatenations );
        auto left_last_prob =
            std::pow( hasher.collision_probability( distance ), last_left_concatenations );

        auto right_prob =
            std::pow( hasher.collision_probability( distance ), cur_right_concatenations );
        auto right_last_prob =
            std::pow( hasher.collision_probability( distance ), last_right_concatenations );

        auto cur_upper_left_prob = 1.0 - std::pow( 1.0 - left_prob, cur_repetitions );
        // auto last_upper_left_prob = 1.0 - std::pow( 1.0 - left_last_prob, cur_repetitions );
        auto last_lower_left_prob = 1.0 - std::pow( 1.0 - left_last_prob, last_repetitions );
        auto cur_upper_right_prob = 1.0 - std::pow( 1.0 - right_prob, cur_repetitions );
        auto last_upper_right_prob = 1.0 - std::pow( 1.0 - right_last_prob, cur_repetitions );
        // auto last_lower_right_prob = 1.0 - std::pow( 1.0 - right_last_prob, last_repetitions );
        // TODO: there are two components commented out down below: including those two makes the
        // failure probability too optimistic
        return ( 1 - cur_upper_left_prob * cur_upper_right_prob ) *
               // ( 1 - last_upper_left_prob * last_upper_right_prob );
               ( 1 - last_lower_left_prob * last_upper_right_prob );
        // ( 1 - last_lower_left_prob * last_lower_right_prob );
    }

    //! Returns the largest distance that attains the given failure probability
    //! at the given concatenations and repetitions, i.e. the inverse of
    //! `failure_probability` in its distance argument.
    //!
    //! Shared by `Index` and `PairForestIndex`: the bisection below depends on
    //! nothing but the hasher, so there is no reason for either index to own a
    //! copy of it.
    template <typename Hasher>
    static float distance_at_failure_probability( Hasher& hasher, float delta, size_t concat,
                                                  size_t rep, size_t max_rep ) {
        // The failure probability is monotonically non-decreasing in the distance:
        // farther pairs have a smaller collision probability and are therefore more
        // likely to be missed. We binary-search for the largest distance whose
        // failure probability does not exceed delta.
        auto fp_at = [&]( float dist ) -> float {
            return failure_probability( hasher, dist, concat, rep, max_rep );
        };

        // A distance leaving the valid domain of the metric yields a non-finite
        // failure probability; we treat such distances as unacceptable so the search
        // stays within the bracket [0, valid).
        auto acceptable = [&]( float dist ) -> bool {
            const float fp = fp_at( dist );
            return std::isfinite( fp ) && fp <= delta;
        };

        // Distance zero collides with probability one, so it never fails. If even
        // that is not acceptable (e.g. delta < 0) there is nothing to return.
        float lo = 0.0f;
        if ( !acceptable( lo ) ) {
            return lo;
        }

        // Grow an upper bound by doubling until its failure probability exceeds delta
        // (or leaves the valid domain). The doubling cap keeps the loop finite.
        float hi = 1.0f;
        for ( size_t doublings = 0; doublings < 64 && acceptable( hi ); doublings++ ) {
            lo = hi;
            hi *= 2.0f;
        }
        if ( acceptable( hi ) ) {
            // Even the largest probed distance stays below delta; return it as the
            // best available lower bound.
            return hi;
        }

        // Binary search maintaining the invariant: lo is acceptable, hi is not.
        for ( size_t iter = 0; iter < 100; iter++ ) {
            const float mid = 0.5f * ( lo + hi );
            if ( mid <= lo || mid >= hi ) {
                break; // converged to the float resolution
            }
            if ( acceptable( mid ) ) {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        return lo;
    }
} // namespace panna
