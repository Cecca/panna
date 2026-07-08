#include <memory>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <sstream>
#include <stdexcept>
#include <variant>
#include <vector>

#include "panna/data.hpp"
#include "panna/distance.hpp"
#include "panna/lsh/crosspolytope.hpp"
#include "panna/lsh/euclidean.hpp"
#include "panna/lsh/lattice.hpp"
#include "panna/lsh/simhash.hpp"
#include "panna/baselines/toc_emst.hpp"
#include "panna/trieindex.hpp"
#include "panna/emst.hpp"
#include "panna/git_version.hpp"
#include "panna/logging.hpp"

namespace nb = nanobind;

struct AbstractIndex {
    virtual void rebuild() = 0;
    virtual void insert( const nb::ndarray<float, nb::c_contig>& vec ) = 0;
    virtual nb::ndarray<uint32_t, nb::numpy, nb::ndim<1>>
    search( const nb::ndarray<float, nb::shape<-1>>& vec, unsigned int k, float recall ) = 0;
    virtual size_t num_points() const = 0;
    virtual size_t num_repetitions() const = 0;
    virtual std::string describe_family() const = 0;
};

template <typename Dataset, typename Hasher, typename Distance>
struct ConcreteIndex final : AbstractIndex {
    panna::Index<Dataset, Hasher, Distance> inner;

    ConcreteIndex( panna::Index<Dataset, Hasher, Distance> inner ): inner( inner ) {
    }

    size_t num_points() const {
        return inner.num_points();
    }

    size_t num_repetitions() const {
        return inner.num_repetitions();
    }

    std::string describe_family() const {
        return inner.describe_family();
    }

    void rebuild() {
        inner.rebuild();
    }

    void insert( const nb::ndarray<float, nb::c_contig>& vec ) {
        // TODO: handle both one and two dimensional vectors
        if ( vec.ndim() == 1 ) {
            inner.insert( vec.data(), vec.data() + vec.shape( 0 ) );
        } else if ( vec.ndim() == 2 ) {
            size_t nrows = vec.shape( 0 );
            size_t dimensionality = vec.shape( 1 );
            float* data = vec.data();
            for ( size_t row = 0; row < nrows; row++ ) {
                float* begin = data + row * dimensionality;
                float* end = data + ( row + 1 ) * dimensionality;
                expect( end - begin == static_cast<ptrdiff_t>( dimensionality ) );
                inner.insert( begin, end );
            }
        }
    }

    nb::ndarray<uint32_t, nb::numpy, nb::ndim<1>>
    search( const nb::ndarray<float, nb::shape<-1>>& vec, unsigned int k, float recall ) {
        std::vector<std::pair<float, uint32_t>> res;
        res.reserve( k + 1 );
        float delta = 1 - recall;
        size_t dimensions = vec.shape( 0 );
        float* data = vec.data();

        inner.search( data, data + dimensions, k, delta, res );
        expect( res.size() == k );

        std::sort( res.begin(), res.end() );
        uint32_t* out_data = new uint32_t[k];

        for ( size_t i = 0; i < k; i++ ) {
            out_data[i] = res[i].second;
        }

        // Delete 'data' when the 'owner' capsule expires
        nb::capsule owner( out_data, []( void* p ) noexcept { delete[] (float*)p; } );

        return nb::ndarray<uint32_t, nb::numpy, nb::ndim<1>>(
            /* data = */ out_data,
            /* shape = */ { k },
            /* owner = */ owner );
    }
};

struct TrieIndex {
    std::unique_ptr<AbstractIndex> inner;

    TrieIndex( size_t dimensions, std::string distance, nb::kwargs kwargs ) {
        size_t repetitions = 64;
        if ( kwargs.contains( "repetitions" ) ) {
            repetitions = nb::cast<size_t>( kwargs["repetitions"] );
        }
        if ( distance == "cosine" ) {
            std::string hasher = "crosspolytope";
            if ( kwargs.contains( "hasher" ) ) {
                hasher = nb::cast<std::string>( kwargs["hasher"] );
            }
            if ( hasher == "crosspolytope" ) {
                using Builder =
                    panna::CrossPolytopeBuilder<3, panna::UnitNormPoints, panna::CosineDistance>;
                Builder builder( dimensions );
                panna::Index<panna::UnitNormPoints, Builder::Output, panna::CosineDistance> index(
                    dimensions, builder, repetitions );
                inner = std::make_unique<
                    ConcreteIndex<panna::UnitNormPoints, Builder::Output, panna::CosineDistance>>(
                    index );
            } else if ( hasher == "simhash" ) {
                using Builder =
                    panna::SimhashBuilder<24, panna::UnitNormPoints, panna::CosineDistance>;
                Builder builder( dimensions );
                panna::Index<panna::UnitNormPoints, Builder::Output, panna::CosineDistance> index(
                    dimensions, builder, repetitions );
                inner = std::make_unique<
                    ConcreteIndex<panna::UnitNormPoints, Builder::Output, panna::CosineDistance>>(
                    index );
            } else {
                throw nb::value_error( "Unsupported hasher for cosine distance. Use either "
                                       "`crosspolytope` or `simhash`" );
            }
        } else if ( distance == "euclidean" ) {
            using Builder = panna::E2LSHBuilder<8, panna::NormedPoints, panna::EuclideanDistance>;
            Builder builder( 0.0, dimensions );
            panna::Index<panna::NormedPoints, Builder::Output, panna::EuclideanDistance> index(
                dimensions, builder, repetitions );
            inner = std::make_unique<
                ConcreteIndex<panna::NormedPoints, Builder::Output, panna::EuclideanDistance>>(
                index );
        } else {
            throw nb::value_error( "Unsupported distance metric" );
        }
    }

    void rebuild() {
        inner->rebuild();
    }

    void insert( const nb::ndarray<float, nb::c_contig>& vec ) {
        inner->insert( vec );
    }

    nb::ndarray<uint32_t, nb::numpy, nb::ndim<1>>
    search( const nb::ndarray<float, nb::shape<-1>>& vec, unsigned int k, float recall ) {
        return inner->search( vec, k, recall );
    }
};

nb::tuple tree_to_pytuple(std::vector<panna::Edge> & tree) {
    size_t tree_size = tree.size();
    float* weights = new float[tree_size];
    uint32_t* ids = new uint32_t[tree_size * 2];
    for ( size_t i = 0; i < tree_size; i++ ) {
        weights[i] = tree[i].weight;
        ids[i * 2] = tree[i].a;
        ids[i * 2 + 1] = tree[i].b;
    }

    nb::capsule weights_owner( weights, []( void* p ) noexcept { delete[] (float*)p; } );
    nb::capsule ids_owner( ids, []( void* p ) noexcept { delete[] (uint32_t*)p; } );

    return nb::make_tuple(
        nb::ndarray<nb::numpy, float, nb::ndim<1>>( weights, { tree_size }, weights_owner ),
        nb::ndarray<nb::numpy, uint32_t, nb::ndim<2>>( ids, { tree_size, 2 }, ids_owner ) );
}

enum EuclideanHashFamily {
    Lattice,
    E2LSH,
    CrossPolytope,
    Simhash,

};

EuclideanHashFamily string_to_family(std::string s) {
    if (s == "lattice") {
        return EuclideanHashFamily::Lattice;
    } else if (s == "e2lsh") {
        return EuclideanHashFamily::E2LSH;
    } else if (s == "crosspolytope") {
        return EuclideanHashFamily::CrossPolytope;
    } else if (s == "simhash") {
        return EuclideanHashFamily::Simhash;
    } else {
        throw std::invalid_argument(
            "only `lattice`, `e2lsh`, `crosspolytope` and `simhash` are supported");
    }
}

struct EMST_exposed {
    using LatticeHasher = panna::LatticeLSH<4, panna::EuclideanPoints, panna::EuclideanDistance>;
    using E2LSHHasher = panna::E2LSH<12, panna::EuclideanPoints, panna::EuclideanDistance>;
    using CrossPolytopeHasher =
        panna::CrossPolytope<4, panna::UnitNormPoints, panna::CosineDistance>;
    using CrossPolytopeEMST =
        panna::EMST<panna::UnitNormPoints, CrossPolytopeHasher, panna::CosineDistance>;
    using SimhashHasher = panna::Simhash<24, panna::UnitNormPoints, panna::CosineDistance>;
    using SimhashEMST =
        panna::EMST<panna::UnitNormPoints, SimhashHasher, panna::CosineDistance>;
    using Variants = std::variant<
        panna::EMST<panna::EuclideanPoints, LatticeHasher, panna::EuclideanDistance>,
        panna::EMST<panna::EuclideanPoints, E2LSHHasher, panna::EuclideanDistance>,
        CrossPolytopeEMST,
        SimhashEMST
    >;

    Variants inner;
    EuclideanHashFamily family;

    // Constructor to be called from Python. It takes a NumPy array and optional keyword arguments.
    EMST_exposed(const nb::ndarray<float, nb::c_contig>& data_in, nb::kwargs kwargs) {
        family = EuclideanHashFamily::Lattice;
        size_t num_points = data_in.shape(0);
        size_t dimensions = data_in.shape(1);

        // Set default parameters, which can be overridden by kwargs from Python
        size_t repetitions = 500;
        double delta = 0.1;
        float epsilon = 0.2;

        if (kwargs.contains("repetitions")) {
            repetitions = nb::cast<size_t>(kwargs["repetitions"]);
        }
        if (kwargs.contains("delta")) {
            delta = nb::cast<double>(kwargs["delta"]);
        }
        if (kwargs.contains("epsilon")) {
            epsilon = nb::cast<float>(kwargs["epsilon"]);
        }

        // Take ownership of the input data
        std::vector<std::vector<float>> data_cpp(num_points, std::vector<float>(dimensions));
        for (size_t i = 0; i < num_points; i++) {
            for (size_t j = 0; j < dimensions; j++) {
                data_cpp[i][j] = data_in.data()[i * dimensions + j];
            }
        }

        if (kwargs.contains("family")) {
            family = string_to_family(nb::cast<std::string>(kwargs["family"]));
        }
        switch ( family ) {
        case Lattice:
            inner
                .emplace<panna::EMST<panna::EuclideanPoints, LatticeHasher, panna::EuclideanDistance>>(
                    dimensions, repetitions, data_cpp, delta, epsilon );
            break;
        case E2LSH:
            inner
                .emplace<panna::EMST<panna::EuclideanPoints, E2LSHHasher, panna::EuclideanDistance>>(
                    dimensions, repetitions, data_cpp, delta, epsilon );
            break;
        case CrossPolytope:
            inner.emplace<CrossPolytopeEMST>(
                dimensions, repetitions, data_cpp, delta, epsilon );
            break;
        case Simhash:
            inner.emplace<SimhashEMST>(
                dimensions, repetitions, data_cpp, delta, epsilon );
            break;
        }
    }

    /// The cosine-based families (crosspolytope, simhash) search the tree
    /// under the cosine distance on unit-normalized points; report the tree
    /// edges with their Euclidean distance instead.
    void reweight_with_euclidean( std::vector<panna::Edge>& tree ) {
        std::visit(
            [&tree]( auto& index ) {
                const auto& dataset = index.table.get_dataset();
                using Dataset = std::decay_t<decltype( dataset )>;
                if constexpr ( std::is_same_v<Dataset, panna::UnitNormPoints> ) {
                    for ( auto& e : tree ) {
                        e.weight =
                            panna::EuclideanDistance::compute( dataset[e.a], dataset[e.b] );
                    }
                }
            },
            inner );
    }

    nb::dict stats() {
        nb::dict ret;
        std::visit(
            [&]( auto& index ) {
                nb::list profile;
                for (auto el : index.get_profile()) {
                    nb::dict elpy;
                    elpy["elapsed_ms"] = el.elapsed_ms;
                    elpy["prefix"] = el.prefix;
                    elpy["repetition"] = el.repetition;
                    elpy["emst_confirmed_weight"] = el.emst_confirmed_weight;
                    elpy["emst_weight_lower_bound"] = el.emst_weight_lower_bound;
                    elpy["emst_max_weight"] = el.emst_max_weight;
                    elpy["emst_max_confirmed_weight"] = el.emst_max_confirmed_weight;
                    elpy["emst_total_weight"] = el.emst_total_weight;
                    elpy["emst_num_confirmed"] = el.emst_num_confirmed;
                    profile.append(elpy);
                }
                ret["distance_count"] = index.get_distance_count();
                ret["collisions_count"] = index.get_collisions_count();
                ret["unique_pairs_estimate"] = index.get_unique_pairs_estimate();
                ret["index_size_bytes"] = index.get_index_size_bytes();
                ret["profile"] = profile;
            },
            inner );
        return ret;
    }

    nb::tuple find_mst() {
        auto tree = std::visit( []( auto& index ) { return index.find_tree().second; }, inner );
        reweight_with_euclidean( tree );
        return tree_to_pytuple( tree );
    }

    nb::tuple find_mst_exact() {
        auto tree = std::visit( []( auto& index ) { return index.exact_tree().second; }, inner );
        reweight_with_euclidean( tree );
        return tree_to_pytuple( tree );
    }

    // Method to find the MST for the reachability and return results as NumPy arrays
    nb::tuple find_mst_dbscan(unsigned int k) {
        // Call the underlying C++ method
        auto result_pair = std::visit(
            [k]( auto& index ) { return index.find_tree_mutual_reachability_distance( k ); },
            inner );

        // 1. MST Edges
        auto& tree_edges_vec = result_pair.first;
        size_t num_edges = tree_edges_vec.size();

        // Create a vector on the heap that will be owned by NumPy
        auto tree_vec_ptr = std::make_unique<std::vector<float>>(num_edges * 3);

        for (size_t i = 0; i < num_edges; ++i) {
            // FIXME: be consistent with fast_hdbscan, i.e. the weight is the last column
            (*tree_vec_ptr)[i * 3 + 0] = tree_edges_vec[i].weight;
            (*tree_vec_ptr)[i * 3 + 1] = tree_edges_vec[i].a;
            (*tree_vec_ptr)[i * 3 + 2] = tree_edges_vec[i].b;
        }

        nb::capsule tree_owner(tree_vec_ptr.get(), [](void *p) noexcept {
            delete static_cast<std::vector<float>*>(p);
        });
        
        nb::ndarray<float, nb::numpy> tree_array(
            tree_vec_ptr.release()->data(), 
            {num_edges, 3},
            tree_owner 
        );
        LOG_INFO("msg", "Created tree array", "num_edges", num_edges);

        // 2. Core Distances
        auto& neighbor_results = result_pair.second;
        size_t num_points = neighbor_results.size();

        auto core_vec_ptr = std::make_unique<std::vector<float>>(num_points);
        for (size_t i = 0; i < num_points; ++i) {
            (*core_vec_ptr)[i] = neighbor_results.core_distance(i);
        }

        nb::capsule core_owner(core_vec_ptr.get(), [](void *p) noexcept {
            delete static_cast<std::vector<float>*>(p);
        });

        LOG_INFO("msg", "Created core array", "num_points", num_points);

        nb::ndarray<float, nb::numpy, nb::ndim<1>> core_array(
            core_vec_ptr.release()->data(),
            {num_points},
            core_owner
        );


        // 3. Neighbors
        size_t num_neighbors_per_point = k + 1;
        auto neighbors_vec_ptr = std::make_unique<std::vector<uint32_t>>(num_points * num_neighbors_per_point);
        
        for (size_t i = 0; i < num_points; ++i) {
            for (size_t j = 0; j < num_neighbors_per_point; ++j) {
                auto nn = neighbor_results.get_neighbors(i);
                if (j < nn.size()) {
                    (*neighbors_vec_ptr)[i * num_neighbors_per_point + j] = nn[j];
                } else {
                    (*neighbors_vec_ptr)[i * num_neighbors_per_point + j] = i;
                }
            }
        }

        nb::capsule neighbors_owner(neighbors_vec_ptr.get(), [](void *p) noexcept {
            delete static_cast<std::vector<uint32_t>*>(p);
        });

        nb::ndarray<uint32_t, nb::numpy, nb::ndim<2>> neighbors_array(
            neighbors_vec_ptr.release()->data(),
            {num_points, num_neighbors_per_point},
            neighbors_owner
        );
        LOG_INFO("msg", "Created neighbors array", "num_points", num_points, "num_neighbors_per_point", num_neighbors_per_point);
        // Return as a python tuple
        return nb::make_tuple(tree_array, core_array, neighbors_array);
    }
};

nb::tuple emst_theory_of_computing( nb::ndarray<float, nb::c_contig>& data_in, nb::kwargs kwargs ) {
    float delta = 0.1;
    float gamma = 1.0;

    if ( kwargs.contains( "delta" ) ) {
        delta = nb::cast<float>( kwargs["delta"] );
    }
    if ( kwargs.contains( "gamma" ) ) {
        gamma = nb::cast<float>( kwargs["gamma"] );
    }

    size_t nrows = data_in.shape( 0 );
    size_t dimensionality = data_in.shape( 1 );
    panna::NormedPoints dataset( dimensionality );
    float* data = data_in.data();
    for ( size_t row = 0; row < nrows; row++ ) {
        float* begin = data + row * dimensionality;
        float* end = data + ( row + 1 ) * dimensionality;
        expect( end - begin == static_cast<ptrdiff_t>( dimensionality ) );
        dataset.push_back( begin, end );
    }
    LOG_INFO("nrows", nrows, "dimensionality", dimensionality);

    using Distance = panna::EuclideanDistance;
    using Hasher = panna::LatticeLSH<4, panna::NormedPoints, Distance>;
    Hasher::Builder builder( dimensionality );

    auto res =
        panna::baselines::emst_theory_of_computing<panna::NormedPoints, Hasher::Builder, Distance>(
            dataset, gamma, delta, builder );
    auto tree = res.second;

    size_t tree_size = tree.size();
    float* weights = new float[tree_size];
    uint32_t* ids = new uint32_t[tree_size * 2];
    for ( size_t i = 0; i < tree_size; i++ ) {
        weights[i] = std::get<0>( tree[i] );
        ids[i * 2] = std::get<1>( tree[i] );
        ids[i * 2 + 1] = std::get<2>( tree[i] );
    }

    nb::capsule weights_owner( weights, []( void* p ) noexcept { delete[] (float*)p; } );
    nb::capsule ids_owner( ids, []( void* p ) noexcept { delete[] (uint32_t*)p; } );

    return nb::make_tuple(
        nb::ndarray<nb::numpy, float, nb::ndim<1>>(weights, {tree_size}, weights_owner),
        nb::ndarray<nb::numpy, uint32_t, nb::ndim<2>>(ids, {tree_size, 2}, ids_owner)
    );
}

// nb::ndarray<float, nb::numpy, nb::ndim<1>>
nb::tuple
distance_histogram( nb::ndarray<float, nb::c_contig>& data_in,
                    // nb::ndarray<float, nb::c_contig, nb::ndim<1>>& bin_bounds,
                    size_t n_bins,
                    float min_distance,
                    float max_distance,
                    size_t sample_size ) {
    size_t nrows = data_in.shape( 0 );
    size_t dimensionality = data_in.shape( 1 );
    panna::NormedPoints dataset( dimensionality );
    float* data = data_in.data();
    for ( size_t row = 0; row < nrows; row++ ) {
        float* begin = data + row * dimensionality;
        float* end = data + ( row + 1 ) * dimensionality;
        expect( end - begin == static_cast<ptrdiff_t>( dimensionality ) );
        dataset.push_back( begin, end );
    }

    // TODO: make the distance configurable
    std::vector<float> counts_vec;
    std::vector<float> bounds_vec;
    std::tie( counts_vec, bounds_vec ) = panna::distance_histogram<panna::EuclideanDistance>(
        dataset, n_bins, min_distance, max_distance, sample_size );

    float* counts = new float[counts_vec.size()];
    for ( size_t i = 0; i < counts_vec.size(); i++ ) {
        counts[i] = counts_vec[i];
    }
    nb::capsule counts_owner( counts, []( void* p ) noexcept { delete[] (float*)p; } );
    const auto counts_numpy = nb::ndarray<float, nb::numpy, nb::ndim<1>>(
        counts, { counts_vec.size() }, counts_owner );

    float* bounds = new float[bounds_vec.size()];
    for ( size_t i = 0; i < bounds_vec.size(); i++ ) {
        bounds[i] = bounds_vec[i];
    }
    nb::capsule bounds_owner( bounds, []( void* p ) noexcept { delete[] (float*)p; } );
    const auto bounds_numpy = nb::ndarray<float, nb::numpy, nb::ndim<1>>(
        bounds, { bounds_vec.size() }, bounds_owner );

    return nb::make_tuple(counts_numpy, bounds_numpy);
}

float
approximate_diameter( nb::ndarray<float, nb::c_contig>& data_in ) {
    size_t nrows = data_in.shape( 0 );
    size_t dimensionality = data_in.shape( 1 );
    panna::NormedPoints dataset( dimensionality );
    float* data = data_in.data();
    for ( size_t row = 0; row < nrows; row++ ) {
        float* begin = data + row * dimensionality;
        float* end = data + ( row + 1 ) * dimensionality;
        expect( end - begin == static_cast<ptrdiff_t>( dimensionality ) );
        dataset.push_back( begin, end );
    }

    float diam = 
        panna::approximate_diameter<panna::EuclideanDistance>( dataset );
    return diam;
}

NB_MODULE( _panna_impl, m ) {
    m.attr( "git_version" ) = GIT_COMMIT_HASH;
    m.def( "set_seed",
           &panna::seed_global_rng,
           "Set the seed of the global random number generato used by the panna module." );
    m.def( "emst_theory_of_computing",
           &emst_theory_of_computing ),
    m.def( "distance_histogram",
           &distance_histogram ),
    m.def( "approximate_diameter",
           &approximate_diameter ),
    nb::class_<TrieIndex>( m, "TrieIndex" )
        .def( nb::init<size_t, std::string, nb::kwargs>() )
        .def( "insert", &TrieIndex::insert )
        .def( "rebuild", &TrieIndex::rebuild )
        .def( "search", &TrieIndex::search )
        .def_prop_ro( "num_repetitions",
                      []( TrieIndex* self ) { return self->inner->num_repetitions(); } )
        .def_prop_ro( "num_points", []( TrieIndex* self ) { return self->inner->num_points(); } )
        .def( "__str__", []( TrieIndex* self ) {
            std::stringstream sstream;
            sstream << "TrieIndex("
                    << "repetitions=" << self->inner->num_repetitions()
                    << " num_points=" << self->inner->num_points()
                    << " family=" << self->inner->describe_family() << ")";
            return sstream.str();
        } );

    nb::class_<EMST_exposed>( m, "EMST")
        .def(nb::init<const nb::ndarray<float, nb::c_contig>&, nb::kwargs>(),
             "Constructs the EMST index from a NumPy array of data points.")
        .def_ro_static("version", &panna::EMST_VERSION)
        // Bind the find_mst method
        .def("find_mst_dbscan", &EMST_exposed::find_mst_dbscan, nb::arg("k") = 5,
            "Find the minimum spanning tree (MST) and the k-NNs for each node.")
        .def("find_mst_exact", &EMST_exposed::find_mst_exact,
             "Find the exact minimum spanning tree (MST) for the dataset.")
        .def("find_mst", &EMST_exposed::find_mst,
             "Find the minimum spanning tree (MST) for the dataset.")
        .def("stats", &EMST_exposed::stats,
             "Return a dictionary with execution-related statistics");

}
