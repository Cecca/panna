#pragma once
#include <cstdint>
#include <numeric>
#include <vector>

// TODO: add the possibility to clear and reuse the data structure
namespace panna {
    /// @brief Disjoint Set Union data structure, implemented with path compression and union by
    /// rank.
    struct DSU {
        std::vector<uint32_t> parent, rank;

        /// @brief Create a Union Find data structure
        /// @param n number of elements
        DSU( uint32_t n ): parent( n ), rank( n, 0 ) {
            std::iota( parent.begin(), parent.end(), 0 );
        }

        size_t size() const {
            return parent.size();
        }

        /// Resets the information in this data structure
        void reset() {
            std::iota( parent.begin(), parent.end(), 0 );
            std::fill( rank.begin(), rank.end(), 0 );
        }

        uint32_t num_connected_components() const {
            size_t cnt = 0;
            for (size_t i=0; i<size(); i++) {
                if (i == cfind(i)) {
                    cnt++;
                }
            }
            return cnt;
        }

        /// @brief Return the parent of the set containing x
        /// @param x, the element to find
        /// @return the parent of the set containing x
        uint32_t find( uint32_t x ) {
            if ( parent.at(x) != x )
                parent.at(x) = find( parent.at(x) ); // Path compression
            return parent.at(x);
        }

        /// Find without path compression, so the operation can be const.
        /// For this to be efficient, paths need to have been compressed
        /// previously by other calls to `find`
        uint32_t cfind( uint32_t x ) const {
            if ( parent.at(x) != x )
                x = cfind( parent.at(x) );
            return x;
        }

        /// Directly access the parent. If the data structure was compressed using
        /// the `compress_all` method, this returns the root without
        /// recursive calls
        uint32_t get_parent( uint32_t x ) const {
            return parent[x];
        }

        /// force the compression of all paths
        void compress_all() {
            for (size_t i=0; i<parent.size(); i++) {
                find(i);
            }
        }

        /// @brief Check if x and y are in the same set
        /// @param x first element
        /// @param y second element
        /// @return true if x and y are in the same set, false otherwise
        bool is_connected( uint32_t x, uint32_t y ) {
            return find( x ) == find( y );
        }

        /// @brief Merge the sets containing x and y in time O(ɑ(n))
        /// @param x first element
        /// @param y second element
        /// @return true if the sets containing x and y were disjoint and were successfully merged,
        /// false otherwise
        bool union_sets( uint32_t x, uint32_t y ) {
            uint32_t rootX = find( x );
            uint32_t rootY = find( y );

            if ( rootX == rootY )
                return false; // Cycle detected

            // Union by rank
            if ( rank.at(rootX) > rank.at(rootY) ) {
                parent.at(rootY) = rootX;
            } else if ( rank.at(rootX) < rank.at(rootY) ) {
                parent.at(rootX) = rootY;
            } else {
                parent.at(rootY) = rootX;
                rank.at(rootX)++;
            }
            return true;
        }
    };

} // namespace panna
