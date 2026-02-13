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

        /// @brief Return the parent of the set containing x
        /// @param x, the element to find
        /// @return the parent of the set containing x
        uint32_t find( uint32_t x ) {
            if ( parent[x] != x )
                parent[x] = find( parent[x] ); // Path compression
            return parent[x];
        }

        /// Find without path compression, so the operation can be const.
        /// For this to be efficient, paths need to have been compressed
        /// previously by other calls to `find`
        uint32_t cfind( uint32_t x ) const {
            while ( parent[x] != x )
                x = parent[x];
            return x;
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
            if ( rank[rootX] > rank[rootY] ) {
                parent[rootY] = rootX;
            } else if ( rank[rootX] < rank[rootY] ) {
                parent[rootX] = rootY;
            } else {
                parent[rootY] = rootX;
                rank[rootX]++;
            }
            // Union by size
            // if (rank.at(rootX) < rank.at(rootY)) {
            //     uint32_t temp = rootX;
            //     rootX = rootY;
            //     rootY = temp;
            // }

            // parent.at(rootY) = rootX;
            // rank.at(rootX) += rank.at(rootY);
            return true;
        }
    };

} // namespace panna
