#pragma once
#include <atomic>
#include <vector>
#include <cstdint>
#include <cmath>
#include <algorithm>
#include <cereal/types/vector.hpp>

namespace panna {

class AtomicBloomFilter {
public:
    AtomicBloomFilter() = default;

    AtomicBloomFilter(size_t size_in_bits, size_t num_hashes) 
        : num_hashes_(num_hashes) {
        // Ensure size is at least 64 bits to avoid empty vector if size_in_bits > 0
        if (size_in_bits > 0) {
            size_t num_words = (size_in_bits + 63) / 64;
            data_.resize(num_words, 0);
            size_ = num_words * 64;
        }
    }

    void resize(size_t size_in_bits, size_t num_hashes) {
        num_hashes_ = num_hashes;
        size_t num_words = (size_in_bits + 63) / 64;
        data_.assign(num_words, 0);
        size_ = num_words * 64;
    }
    
    bool empty() const {
        return data_.empty();
    }

    void clear() {
        std::fill(data_.begin(), data_.end(), 0);
    }
    
    void add(uint32_t a, uint32_t b) {
        if (data_.empty()) return;
        if (a > b) std::swap(a, b);
        uint64_t h1 = hash1(a, b);
        uint64_t h2 = hash2(a, b);
        
        uint64_t bit_idx = h1 % size_;
        uint64_t step = h2 % size_;

        for (size_t i = 0; i < num_hashes_; ++i) {
             set_bit(bit_idx);
             bit_idx += step;
             if (bit_idx >= size_) bit_idx -= size_;
        }
    }

    bool contains(uint32_t a, uint32_t b) const {
        if (data_.empty()) return false;
        if (a > b) std::swap(a, b);
        uint64_t h1 = hash1(a, b);
        uint64_t h2 = hash2(a, b);

        uint64_t bit_idx = h1 % size_;
        uint64_t step = h2 % size_;

        for (size_t i = 0; i < num_hashes_; ++i) {
             if (!test_bit(bit_idx)) return false;
             bit_idx += step;
             if (bit_idx >= size_) bit_idx -= size_;
        }
        return true;
    }

    template <typename Archive>
    void serialize( Archive& ar ) {
        ar( size_, num_hashes_, data_ );
    }

private:
    size_t size_ = 0;
    size_t num_hashes_ = 0;
    mutable std::vector<uint64_t> data_;

    void set_bit(size_t idx) {
        size_t word_idx = idx / 64;
        uint64_t mask = 1ULL << (idx % 64);
        std::atomic_ref<uint64_t> word(data_[word_idx]);
        if (word.load(std::memory_order_relaxed) & mask) return;
        word.fetch_or(mask, std::memory_order_relaxed);
    }

    bool test_bit(size_t idx) const {
        size_t word_idx = idx / 64;
        uint64_t mask = 1ULL << (idx % 64);
        std::atomic_ref<uint64_t> word(data_[word_idx]); 
        return (word.load(std::memory_order_relaxed) & mask) != 0;
    }

     static uint64_t hash1(uint32_t a, uint32_t b) {
         uint64_t k = (static_cast<uint64_t>(a) << 32) | b;
         k ^= k >> 33;
         k *= 0xff51afd7ed558ccdULL;
         k ^= k >> 33;
         k *= 0xc4ceb9fe1a85ec53ULL;
         k ^= k >> 33;
         return k;
    }
    
    static uint64_t hash2(uint32_t a, uint32_t b) {
         uint64_t k = (static_cast<uint64_t>(b) << 32) | a;
         k ^= k >> 33;
         k *= 0xff51afd7ed558ccdULL;
         k ^= k >> 33;
         k *= 0xc4ceb9fe1a85ec53ULL;
         k ^= k >> 33;
         return k;
    }

};
}
