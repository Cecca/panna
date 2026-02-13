#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace panna {
    // Helper function for getting indices to tensor.
    //
    // Retrieve the pair of indices where both sides are incremented as little
    // as possible. The rhs index is incremented first Eg. (0, 0) (0, 1) (1, 0)
    // (1, 1) (0, 2)
    static std::pair<unsigned int, unsigned int> get_minimal_index_pair( int idx ) {
        int sqrt = static_cast<int>( std::sqrt( idx ) );
        if ( idx == sqrt * sqrt + 2 * sqrt ) {
            return { sqrt, sqrt };
        } else if ( idx >= sqrt * sqrt + sqrt ) {
            return { sqrt, idx - ( sqrt * sqrt + sqrt ) };
        } else { // idx >= sqrt*sqrt, always true
            return { idx - sqrt * sqrt, sqrt };
        }
    }

    template <typename Hasher, typename Dataset>
    class Tensoring {
    public:
        using Value = typename Hasher::Value::DoubleWidth;

    private:
        size_t repetitions;
        Hasher inner;

    public:
        template <typename HasherBuilder>
        Tensoring( HasherBuilder builder, size_t repetitions ):
            repetitions( repetitions ), inner( builder.build( repetitions ) ) {
        }

        static constexpr size_t get_concatenations() {
            return 2 * Hasher::get_concatenations();
        }

        size_t get_repetitions() const {
            return repetitions;
        }

        void hash( typename Dataset::PointHandle point, std::vector<Value>& output ) {
            using HalfValue = typename Hasher::Value;
            output.clear();
            output.resize( repetitions );
            // TODO: reuse this allocation
            std::vector<typename Hasher::Value> scratch;
            inner.hash( point, scratch );
            size_t tensored_hashers = scratch.size();
            size_t right_start = tensored_hashers / 2;

            for ( size_t rep = 0; rep < repetitions; rep++ ) {
                auto index_pair = get_minimal_index_pair( rep );
                HalfValue h_left = scratch[index_pair.first];
                HalfValue h_right = scratch[right_start + index_pair.second];
                output[rep] = Value::interleave( h_left, h_right );
            }
            // hide the scratch space
            output.resize( repetitions );
        }

        float collision_probability( float distance ) const {
            return inner.collision_probability( distance );
        }
    };

    template <typename InnerBuilder, typename Dataset>
    class TensoringBuilder {
        InnerBuilder inner;

    public:
        using Output = Tensoring<typename InnerBuilder::Output, Dataset>;

        TensoringBuilder( InnerBuilder inner ): inner( inner ) {
        }

        template <typename InputPoints>
        void fit( InputPoints& input_points ) {
            inner.fit(input_points);
        }

        Output build( size_t repetitions ) const {
            return Tensoring<typename InnerBuilder::Output, Dataset>( inner, repetitions );
        }
    };
} // namespace panna
