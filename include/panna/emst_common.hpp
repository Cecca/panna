#pragma once

//! Pieces of the EMST search that do not depend on any particular index.
//!
//! `emst.hpp` drives `panna::Index`/`PrefixMap` through a persistent thread
//! pool; `pairemst.hpp` drives `PairForestIndex` through OpenMP. Both need the
//! same Kruskal primitives, the same seeding trees and the same stopping rule,
//! and neither of those needs a `Billboard`, a `Channel` or a `trieindex.hpp`.
//! Keeping them here is what lets `pairemst.hpp` stay clear of that machinery.
//!
//! Everything in this header was extracted verbatim from `emst.hpp`, with the
//! single exception of the free `stopping_condition` below, which is the body
//! of `EMST::stopping_condition` parameterised on the confirmed distance
//! instead of reaching into an `Index`.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <utility>
#include <vector>

#include "panna/data.hpp"
#include "panna/dsu.hpp"
#include "panna/expect.hpp"
#include "panna/logging.hpp"
#include "panna/rand.hpp"
#include "panna/timer.hpp"

namespace panna {
    // All weights are euclidean distances, even when the Distance
    // template argument is something else. Conversion is via the
    // to_euclidean method
    struct StoppingConditionInfo {
        const float total_weight;
        const float confirmed_weight;
        const float heaviest_confirmed_edge;
        const size_t edges_to_confirm;
        const size_t confirmed_edges;
    };

    template <typename Edge>
    static void kruskal( DSU& dsu, std::vector<Edge>& edge_list, std::vector<Edge>& output ) {
        for ( const auto& edge : edge_list ) {
            if ( output.size() == dsu.size() - 1 ) {
                break;
            }
            if ( dsu.union_sets( edge.a, edge.b ) ) {
                output.push_back( edge );
            }
        }
    }

    template <typename Dataset, typename Distance>
    std::vector<Edge> random_emst( const Dataset& data ) {
        Timer _t("random_emst");
        const size_t num_data = data.size();
        std::vector<Edge> edges;
        const size_t samples = num_data * std::ceil( std::log10( num_data ) );
        edges.reserve( samples );
        for ( size_t i = 0; i < samples; i++ ) {
            const size_t a = sample_int( 0, num_data - 1 );
            const size_t b = sample_int( 0, num_data - 1 );
            const float d = Distance::compute( data[a], data[b] );
            edges.emplace_back( d, a, b );
        }
        std::sort( edges.begin(), edges.end() );
        std::vector<Edge> res;
        res.reserve( num_data - 1 );
        DSU dsu( num_data );
        kruskal( dsu, edges, res );
        const uint32_t root = 0;
        while ( res.size() < num_data - 1 ) {
            // add arbitrary edges
            for (size_t i=0; i<num_data; i++) {
                if (dsu.union_sets(root, i)) {
                    const float weight = Distance::compute(data[root], data[i]);
                    res.emplace_back(weight, root, i);
                }
            }
        }
        std::sort( res.begin(), res.end() );
        expect( res.size() == num_data - 1 );
        return res;
    }

    template<typename Dataset, typename Distance>
    static std::pair<float, std::vector<Edge>> exact_emst( const Dataset& data ) {
        Timer _t("exact_emst");
        // Compute all the distances
        //  We can pre-allocate all the memory, and avoid the critical region
        const size_t num_data = data.size();
        std::vector<Edge> all_edges( ( num_data - 1 ) * num_data / 2 );
#pragma omp parallel for collapse( 2 )
        for ( size_t i = 0; i < num_data; i++ ) {
            for ( size_t j = i + 1; j < num_data; j++ ) {
                float dist = Distance::compute( data[i], data[j] );
                all_edges.at( i * ( num_data - 1 ) - ( i * ( i + 1 ) / 2 ) + j - 1 ) =
                    Edge{ .weight = dist, .a = (uint32_t)i, .b = (uint32_t)j };
            }
        }
        // Sort the edges
        std::sort( all_edges.begin(), all_edges.end() );
        // Create the DSU
        DSU dsu( num_data );
        float tree_weight = 0;
        std::cout << "Creating the MST" << std::endl;
        std::vector<Edge> tree;
        kruskal( dsu, all_edges, tree );
        expect( tree.size() > 0 );
        LOG_INFO( "msg", "MST created", "heaviest_edge", tree.back().weight );
        for ( const auto& edge : tree ) {
            tree_weight += edge.weight;
        }
        return { tree_weight, tree };
    }

    /// Builds a spanning tree as follows. First the data points are clustered in
    /// std::sqrt(data.size()) clusters with the kcenter algorithm. Then, we compute
    /// the exact EMST of the cluster centers. Finally, we add, for each non-center
    /// point, the edge between itself and its closest cluster center.
    template <typename Dataset, typename Distance>
    std::vector<Edge> clustering_emst( const Dataset& data ) {
        Timer _t("clustering_emst");
        const size_t num_data = data.size();
        const size_t num_clusters = std::ceil( std::sqrt( num_data ) );
        const auto clustering = kcenter<Distance>( data, num_clusters );

        std::vector<Edge> res;
        res.reserve( num_data - 1 );

        // the exact EMST of the centers, with the edge endpoints remapped
        // to the indices of the centers in the original dataset
        const auto [centers_weight, centers_tree] =
            exact_emst<Dataset, Distance>( clustering.centers );
        for ( const auto& edge : centers_tree ) {
            const uint32_t ida = (uint32_t)clustering.center_indices.at( edge.a );
            const uint32_t idb = (uint32_t)clustering.center_indices.at( edge.b );
            if(ida == idb) {
                throw std::runtime_error( "invalid edge!" );
            }
            res.emplace_back( edge.weight, ida, idb );
        }

        // connect each non-center point to its closest center
        std::vector<bool> is_center( num_data, false );
        for ( const size_t c : clustering.center_indices ) {
            is_center.at( c ) = true;
        }
        for ( size_t i = 0; i < num_data; i++ ) {
            if ( !is_center.at( i ) ) {
                res.emplace_back(
                    clustering.distances.at( i ),
                    (uint32_t)i,
                    (uint32_t)clustering.center_indices.at( clustering.assignment.at( i ) ) );
            }
        }

        std::sort( res.begin(), res.end() );
        expect( res.size() == num_data - 1 );
        return res;
    }

    /// `weights` must be sorted in ascending order
    static std::vector<float> find_breaks( const std::vector<float>& weights, float step ) {
        std::vector<float> breaks;
        breaks.push_back( weights.back() );
        LOG_INFO( "weight-break-point", breaks.back() );
        for ( int32_t i = weights.size() - 1; i >= 0; i-- ) {
            const float w = weights[i];
            if (w == 0.0) {
                break;
            }
            if ( w < breaks.back() / step ) {
                LOG_INFO( "weight-break-point", w );
                breaks.push_back( w );
            }
        }
        std::reverse(breaks.begin(), breaks.end());
        return breaks;
    }

    static std::vector<float> find_breaks( const std::vector<Edge>& tree, float step ) {
        std::vector<float> weights;
        weights.reserve( tree.size() );
        for ( const auto& e : tree ) {
            weights.push_back( e.weight );
        }
        return find_breaks( weights, step );
    }

    /// Simulate a run of Kruskal's algorithm, assuming both input vectors are sorted.
    /// Report in the output vector the edges from `new_edges` that would be
    /// part of the updated tree.
    static void kruskal_new_edges( const std::vector<Edge>& old_edges,
                                   const std::vector<Edge>& new_edges,
                                   DSU& union_find,
                                   std::vector<Edge>& out ) {
        expect( std::is_sorted( old_edges.begin(), old_edges.end() ) );
        expect( std::is_sorted( new_edges.begin(), new_edges.end() ) );

        union_find.reset();
        size_t asize = old_edges.size();
        size_t bsize = new_edges.size();
        size_t aidx = 0;
        size_t bidx = 0;

        while ( aidx < asize && bidx < bsize ) {
            if ( old_edges.at( aidx ) < new_edges.at( bidx ) ) {
                auto e = old_edges.at(aidx++);
                union_find.union_sets( e.a, e.b );
            } else {
                auto e = new_edges.at(bidx++);
                if ( union_find.union_sets( e.a, e.b ) ) {
                    out.push_back( e );
                }
            }
        }
        while ( aidx < asize ) {
            auto e = old_edges.at(aidx++);
            union_find.union_sets( e.a, e.b );
        }
        while ( bidx < bsize ) {
            auto e = new_edges.at(bidx++);
            if ( union_find.union_sets( e.a, e.b ) ) {
                out.push_back( e );
            }
        }
    }


    /// implementation of Kruskal's algorithm that picks updates from two sorted
    /// vectors. Avoids having to sort both their concatenation.
    static void kruskal_merge( const std::vector<Edge>& old_edges,
                                   const std::vector<Edge>& new_edges,
                                   DSU& union_find,
                                   std::vector<Edge>& out ) {
        expect( std::is_sorted( old_edges.begin(), old_edges.end() ) );
        expect( std::is_sorted( new_edges.begin(), new_edges.end() ) );

        union_find.reset();
        size_t asize = old_edges.size();
        size_t bsize = new_edges.size();
        size_t aidx = 0;
        size_t bidx = 0;

        while ( aidx < asize && bidx < bsize ) {
            Edge e;
            if ( old_edges.at( aidx ) < new_edges.at( bidx ) ) {
                e = old_edges.at(aidx++);
            } else {
                e = new_edges.at(bidx++);
            }
            if ( union_find.union_sets( e.a, e.b ) ) {
                out.push_back( e );
            }
        }
        while ( aidx < asize ) {
            auto e = old_edges.at(aidx++);
            if ( union_find.union_sets( e.a, e.b ) ) {
                out.push_back( e );
            }
        }
        while ( bidx < bsize ) {
            auto e = new_edges.at(bidx++);
            if ( union_find.union_sets( e.a, e.b ) ) {
                out.push_back( e );
            }
        }
    }

    //! The stopping rule of every EMST search in this codebase, split off from
    //! the index that supplies `confirmed_distance`.
    //!
    //! `tree` must be a spanning tree sorted by ascending weight. Its prefix of
    //! edges no heavier than `confirmed_distance` is *confirmed*: with
    //! probability at least `1 - delta` the LSH sweep has already seen every
    //! pair at that distance, so those edges are final. The remaining
    //! `edges_to_confirm` edges are not, but each of them is at least as heavy
    //! as the heaviest confirmed one, which is what turns
    //! `confirmed_weight + edges_to_confirm * heaviest_confirmed_edge`
    //! into a lower bound on the true MST weight.
    //!
    //! All the weights reported here are euclidean distances, converted through
    //! `Distance::to_euclidean`, so that the caller's epsilon applies to the
    //! quantity the approximation guarantee is stated about.
    template <typename Distance>
    static StoppingConditionInfo stopping_condition( const std::vector<Edge>& tree,
                                                     float confirmed_distance ) {
        float weight = 0.0f;
        size_t idx = 0;
        while ( idx < tree.size() ) {
            const float w = tree.at( idx ).weight;
            if ( w > confirmed_distance ) {
                break;
            }
            weight += Distance::to_euclidean( w );
            idx += 1;
        }

        size_t edges_to_confirm = tree.size() - idx;

        float total_weight = weight;
        for ( size_t jj = idx; jj < tree.size(); jj++ ) {
            float w = tree.at( jj ).weight;
            total_weight += Distance::to_euclidean( w );
        }

        float heaviest = ( idx > 0 ) ? Distance::to_euclidean( tree.at( idx - 1 ).weight ) : 0.0f;

        // All distances reported here are euclidean, so that
        // the epsilon for the approximation is applied correctly
        return StoppingConditionInfo{ .total_weight = total_weight,
                                      .confirmed_weight = weight,
                                      .heaviest_confirmed_edge = heaviest,
                                      .edges_to_confirm = edges_to_confirm,
                                      .confirmed_edges = idx };
    }

} // namespace panna
