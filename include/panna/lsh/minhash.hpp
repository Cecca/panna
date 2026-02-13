#pragma once

#include <cstdint>
#include <limits>
#include <random>

#include "panna/lsh/values.hpp"
#include "panna/rand.hpp"
namespace panna {

    class TabulationHash {
        uint64_t t1[256];
        uint64_t t2[256];
        uint64_t t3[256];
        uint64_t t4[256];

    public:
        TabulationHash() {
            auto& rng = get_global_rng();
            for ( size_t i = 0; i < 256; i++ ) {
                t1[i] = rng();
                t2[i] = rng();
                t3[i] = rng();
                t4[i] = rng();
            }
        }

        uint64_t operator()( uint32_t val ) const {
            return ( t1[val & 0xFF] ^ t2[( val >> 8 ) & 0xFF] ^
                     t3[( val >> 16 ) & 0xFF] ^ t4[( val >> 24 ) & 0xFF] );
        }
    };

    template <uint8_t K, typename Dataset>
    class MinhashBuilder;

    template <uint8_t K, typename Dataset>
    class Minhash {
        size_t repetitions;
        std::vector<TabulationHash> hashes;

    public:
        using Value = IntLshValue<K>;
        using Builder = MinhashBuilder<K, Dataset>;

        Minhash( size_t repetitions ): repetitions( repetitions ) {
            for ( size_t i=0; i < repetitions * K; i++ ) {
                hashes.push_back( TabulationHash() );
            }
        }

        static constexpr size_t get_concatenations() { return K; }

        uint32_t hash_single( typename Dataset::PointHandle set,
                              size_t concatenation,
                              size_t repetition ) const {
            TabulationHash hash = hashes[repetition * K + concatenation];
            uint64_t min_hash = std::numeric_limits<uint64_t>::max();
            uint32_t min_token = 0;
            for ( const uint32_t* it = set.begin(); it != set.end(); it++ ) {
                uint64_t h = hash( *it );
                if ( h < min_hash ) {
                    min_hash = h;
                    min_token = *it;
                }
            }
            return min_token;
        }

        void hash( typename Dataset::PointHandle point,
                   std::vector<Value>& output ) {
            // OPTIMIZE: TabulationHasher gives 64 bit hashes, hence we could use
            // its output to hash two repetitions/concatenations
            output.clear();
            Value cur;
            for ( size_t rep = 0; rep < repetitions; rep++ ) {
                for ( size_t concat = 0; concat < K; concat++ ) {
                    int32_t code = hash_single( point, concat, rep );
                    cur.set( concat, code );
                }
                output.push_back( cur );
                cur = Value();
            }
        }

        float collision_probability(float distance) const {
            return 1 - distance;
        }
    };

    template <uint8_t K, typename Dataset>
    class MinhashBuilder {
    public:
        using Output = Minhash<K, Dataset>;

        MinhashBuilder() {}

        template <typename Archive>
        void serialize( Archive& ar ) {
            ar();
        }

        void fit( Dataset& ) {
        }

        Output build( size_t repetitions ) const {
            return Minhash<K, Dataset>( repetitions );
        }

        std::string describe() const {
            return "MinHash1Bit";
        }
    };

    template <uint8_t K, typename Dataset>
    class Minhash1BitBuilder;

    template <uint8_t K, typename Dataset>
    class Minhash1Bit {
        size_t repetitions;
        Minhash<K, Dataset> minhash;

    public:
        using Value = BitwiseLshValue<K>;
        using Builder = Minhash1BitBuilder<K, Dataset>;

        Minhash1Bit( size_t repetitions ): repetitions(repetitions), minhash( repetitions ) {
        }

        template <typename Archive>
        void serialize( Archive& ar ) {
            ar();
        }

        void hash( typename Dataset::PointHandle point,
                   std::vector<Value>& output ) {
            // OPTIMIZE: TabulationHasher gives 64 bit hashes, hence we could use
            // its output to hash two repetitions/concatenations
            output.clear();
            uint32_t cur;
            for ( size_t rep = 0; rep < repetitions; rep++ ) {
                for ( size_t concat = 0; concat < K; concat++ ) {
                    int32_t code = minhash.hash_single( point, concat, rep ) % 2;
                    cur = (cur << 1) | code;
                }
                output.push_back( Value::make(cur));
                cur = 0;
            }
        }

        float collision_probability(float distance) const {
            float similarity = 1 - distance;
            return (1 + similarity) / 2;
        }
    };

    template <uint8_t K, typename Dataset>
    class Minhash1BitBuilder {
    public:
        using Output = Minhash1Bit<K, Dataset>;

        Minhash1BitBuilder() {}

        void fit( Dataset& ) {
        }

        Output build( size_t repetitions ) const {
            return Minhash1Bit<K, Dataset>( repetitions );
        }

        std::string describe() const {
            return "MinHash1Bit";
        }
    };
} // namespace panna
