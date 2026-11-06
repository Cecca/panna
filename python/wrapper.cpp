#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/unique_ptr.h>
#include <omp.h>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include "panna/data.hpp"
#include "panna/distance.hpp"
#include "panna/lsh/crosspolytope.hpp"
#include "panna/lsh/euclidean.hpp"
#include "panna/lsh/lattice.hpp"
#include "panna/lsh/minhash.hpp"
#include "panna/lsh/simhash.hpp"
#include "panna/baselines/toc_emst.hpp"
#include "panna/trieindex.hpp"
#include "panna/emst.hpp"
#include "panna/pairemst.hpp"
#include "panna/git_version.hpp"
#include "panna/logging.hpp"

namespace nb = nanobind;

// ---------------------------------------------------------------------------
// Binding helpers for the LSH families
//
// These helpers are shared by the Python classes wrapping the hash families
// of `include/panna/lsh/`. A family wrapper holds a `std::variant` over the
// supported instantiations of `K` (see `KList` and `HasherVariant`), converts
// its input with `to_dataset` or `from_csr`, and hashes it with
// `hash_dataset`, which runs `hash_rows` with the GIL released.
// ---------------------------------------------------------------------------

/// The float arrays accepted by the LSH families. nanobind converts the dtype
/// and the memory layout implicitly when needed.
using FloatArray = nb::ndarray<const float, nb::c_contig, nb::device::cpu>;

/// The integer arrays accepted for CSR input. Integers are taken as `int64`,
/// so that negative values can be detected rather than silently wrapped.
using IndexArray = nb::ndarray<const int64_t, nb::ndim<1>, nb::c_contig, nb::device::cpu>;

/// A compile-time list of the supported concatenation counts `K` of a family.
///
/// `KList` stores nothing at runtime: it is a type that carries the values
/// `Ks...` as a template parameter pack, so that the helpers below can
/// "unpack" them with partial specialization (`HasherVariant`, `DispatchK`)
/// and expand them with pack expansions and fold expressions. For background:
/// - parameter packs: https://en.cppreference.com/w/cpp/language/pack
/// - fold expressions: https://en.cppreference.com/w/cpp/language/fold
/// - partial specialization: https://en.cppreference.com/w/cpp/language/partial_specialization
template <uint8_t... Ks>
struct KList {
    static_assert( sizeof...( Ks ) > 0, "a KList needs at least one value" );

    static constexpr uint8_t first = std::array<uint8_t, sizeof...( Ks )>{ Ks... }[0];

    /// The allowed values, as a comma-separated string.
    static std::string describe() {
        std::stringstream sstream;
        bool is_first = true;
        ( ( sstream << ( is_first ? "" : ", " ) << +Ks, is_first = false ), ... );
        return sstream.str();
    }
};

/// `std::variant<Hasher<K>...>` over the values `K` of a `KList`. `Hasher` is
/// an alias template that fixes the other parameters of the family, for
/// example `template <uint8_t K> using H = panna::Simhash<K, Dataset, Distance>`.
///
/// The primary template is only declared; the partial specialization matches
/// `KList<Ks...>`, which binds the pack `Ks`, and `Hasher<Ks>...` expands it,
/// e.g. `HasherVariant<H, KList<1, 2>>::type` is `std::variant<H<1>, H<2>>`.
template <template <uint8_t> class Hasher, typename List>
struct HasherVariant;

template <template <uint8_t> class Hasher, uint8_t... Ks>
struct HasherVariant<Hasher, KList<Ks...>> {
    using type = std::variant<Hasher<Ks>...>;
};

/// Implementation of `dispatch_k`. It maps the runtime `k` to a compile-time
/// `K`: the `||` fold expression expands to one comparison `k == K` per value
/// of the pack, in order, and short-circuits at the first match, which calls
/// `f` with `std::integral_constant<uint8_t, K>`. Inside `f`, `decltype(kc)::value`
/// is a constant expression, so it can be used as a template argument. All
/// the calls to `f` must return the same type `R` (e.g. a `HasherVariant`).
/// The result goes through a `std::optional<R>`, so `R` need not be
/// default-constructible; a `void` `f` is handled separately.
template <typename List>
struct DispatchK;

template <uint8_t... Ks>
struct DispatchK<KList<Ks...>> {
    template <typename F>
    static auto run( size_t k, F&& f ) {
        using R = std::invoke_result_t<F&, std::integral_constant<uint8_t, KList<Ks...>::first>>;
        if constexpr ( std::is_void_v<R> ) {
            const bool found =
                ( ( k == Ks ? ( f( std::integral_constant<uint8_t, Ks>{} ), true ) : false ) ||
                  ... );
            if ( !found ) {
                unsupported( k );
            }
        } else {
            std::optional<R> out;
            (void)( ( k == Ks ? ( out.emplace( f( std::integral_constant<uint8_t, Ks>{} ) ), true )
                              : false ) ||
                    ... );
            if ( !out ) {
                unsupported( k );
            }
            return std::move( *out );
        }
    }

    [[noreturn]] static void unsupported( size_t k ) {
        std::stringstream sstream;
        sstream << "unsupported k=" << k << "; allowed values are " << KList<Ks...>::describe();
        throw nb::value_error( sstream.str().c_str() );
    }
};

/// Calls `f(std::integral_constant<uint8_t, K>{})` for the `K` of `List` equal
/// to `k`, and returns its result. Raises `ValueError`, listing the allowed
/// values, if `k` is not in `List`.
template <typename List, typename F>
auto dispatch_k( size_t k, F&& f ) {
    return DispatchK<List>::run( k, std::forward<F>( f ) );
}

/// Re-seeds the module-wide global RNG when a seed is given. Note that this
/// affects every later random draw of the module, not only the next one.
/// Call it with `global_rng_mutex` held (see `with_global_rng`).
void maybe_seed( std::optional<uint64_t> seed ) {
    if ( seed.has_value() ) {
        panna::seed_global_rng( *seed );
    }
}

/// Serializes the draws from the module-wide global RNG made by the LSH
/// family classes (sampling and fitting hashers) and by `set_seed`, which run
/// with the GIL released. Other users of the global RNG (`EMST`, `TrieIndex`,
/// ...) hold the GIL instead and do not take this mutex.
std::mutex global_rng_mutex;

/// Runs `f()` with the GIL released and `global_rng_mutex` held, after
/// re-seeding the global RNG if `seed` is given, and returns its result. `f`
/// must not touch Python objects. The GIL is released before the mutex is
/// taken, so that a thread waiting for the mutex never holds the GIL.
template <typename F>
auto with_global_rng( std::optional<uint64_t> seed, F&& f ) {
    nb::gil_scoped_release release;
    std::lock_guard<std::mutex> lock( global_rng_mutex );
    maybe_seed( seed );
    return f();
}

/// The shortest representation of `x` that round-trips through `float`,
/// formatted like a Python float (with a `.0` for integral values).
inline std::string float_repr( float x ) {
    char buffer[64];
    auto [end, ec] = std::to_chars( buffer, buffer + sizeof( buffer ), x );
    std::string out( buffer, ec == std::errc() ? end : buffer );
    if ( out.find_first_of( ".en" ) == std::string::npos ) {
        out += ".0";
    }
    return out;
}

/// Raises `ValueError` unless `value` is positive.
inline void check_positive( size_t value, const char* name ) {
    if ( value == 0 ) {
        throw nb::value_error( ( std::string( name ) + " must be positive" ).c_str() );
    }
}

/// Raises `ValueError` unless `value` is positive and finite.
inline void check_positive( float value, const char* name ) {
    if ( !( value > 0.0f ) || !std::isfinite( value ) ) {
        throw nb::value_error( ( std::string( name ) + " must be positive and finite" ).c_str() );
    }
}

/// Raises `ValueError` unless `delta` is in (0, 1).
inline void check_delta( float delta ) {
    if ( !( delta > 0.0f && delta < 1.0f ) ) {
        throw nb::value_error( "delta must be in (0, 1)" );
    }
}

/// Copies a 1-D (one point) or 2-D (one point per row) C-contiguous float
/// array into a new `Dataset` of the given dimensions. Raises `ValueError` if
/// the array has another number of dimensions, or if its last axis does not
/// match `dimensions`.
template <typename Dataset, typename Array>
Dataset to_dataset( const Array& arr, size_t dimensions ) {
    if ( arr.ndim() != 1 && arr.ndim() != 2 ) {
        std::stringstream sstream;
        sstream << "expected a one- or two-dimensional array, got " << arr.ndim()
                << " dimensions";
        throw nb::value_error( sstream.str().c_str() );
    }
    if ( dimensions == 0 ) {
        throw nb::value_error( "the points must have at least one dimension" );
    }
    const size_t cols = arr.shape( arr.ndim() - 1 );
    if ( cols != dimensions ) {
        std::stringstream sstream;
        sstream << "dimension mismatch: expected points with " << dimensions
                << " dimensions, got " << cols;
        throw nb::value_error( sstream.str().c_str() );
    }
    const size_t rows = arr.ndim() == 1 ? 1 : arr.shape( 0 );
    Dataset dataset( dimensions );
    const float* data = arr.data();
    for ( size_t row = 0; row < rows; row++ ) {
        dataset.push_back( data + row * dimensions, data + ( row + 1 ) * dimensions );
    }
    return dataset;
}

/// As above, taking the dimensions from the last axis of the array.
template <typename Dataset, typename Array>
Dataset to_dataset( const Array& arr ) {
    if ( arr.ndim() == 0 ) {
        throw nb::value_error( "expected a one- or two-dimensional array, got a scalar" );
    }
    return to_dataset<Dataset>( arr, arr.shape( arr.ndim() - 1 ) );
}

/// Builds the sets of a CSR structure: set `i` holds the tokens
/// `indices[indptr[i]:indptr[i+1]]`. Raises `ValueError` unless `indptr` is
/// non-empty, starts at 0, is non-decreasing, and ends at `len(indices)`, and
/// unless every token is in `[0, 2**32)`. Repeated tokens are kept as they are,
/// which does not change the MinHash values.
panna::SparseSets from_csr( const IndexArray& indptr, const IndexArray& indices ) {
    const size_t num_ptrs = indptr.shape( 0 );
    const size_t num_tokens = indices.shape( 0 );
    const int64_t* ptr = indptr.data();
    const int64_t* tok = indices.data();
    if ( num_ptrs == 0 ) {
        throw nb::value_error( "indptr must have at least one element" );
    }
    if ( ptr[0] != 0 ) {
        throw nb::value_error( "indptr must start at 0" );
    }
    for ( size_t i = 1; i < num_ptrs; i++ ) {
        if ( ptr[i] < ptr[i - 1] ) {
            throw nb::value_error( "indptr must be non-decreasing" );
        }
    }
    if ( static_cast<uint64_t>( ptr[num_ptrs - 1] ) != num_tokens ) {
        std::stringstream sstream;
        sstream << "indptr[-1] (" << ptr[num_ptrs - 1] << ") must equal len(indices) ("
                << num_tokens << ")";
        throw nb::value_error( sstream.str().c_str() );
    }
    for ( size_t i = 0; i < num_tokens; i++ ) {
        if ( tok[i] < 0 || tok[i] > static_cast<int64_t>( std::numeric_limits<uint32_t>::max() ) ) {
            throw nb::value_error( "the tokens in indices must be in [0, 2**32)" );
        }
    }

    // `SparseSets` rejects tokens larger than its universe size.
    panna::SparseSets sets( std::numeric_limits<uint32_t>::max() );
    std::vector<uint32_t> set;
    for ( size_t i = 0; i + 1 < num_ptrs; i++ ) {
        set.assign( tok + ptr[i], tok + ptr[i + 1] );
        sets.push_back( set );
    }
    return sets;
}

/// Allocates a NumPy array of the given shape, owned by a capsule that frees
/// it. Returns the raw pointer, to fill the array, along with the array. The
/// contents are uninitialized.
template <typename T>
std::pair<T*, nb::ndarray<nb::numpy, T>> make_owned_array( const std::vector<size_t>& shape ) {
    size_t total = 1;
    for ( size_t s : shape ) {
        total *= s;
    }
    std::unique_ptr<T[]> buffer( new T[total] );
    nb::capsule owner( buffer.get(), []( void* p ) noexcept { delete[] static_cast<T*>( p ); } );
    T* data = buffer.release();
    return { data, nb::ndarray<nb::numpy, T>( data, shape.size(), shape.data(), owner ) };
}

/// Hashers whose `hash` can be called on a `const` object. The others
/// (`CrossPolytope`, `LatticeLSH`) use per-thread scratch space.
template <typename Hasher, typename Dataset>
concept ConstHasher = requires( const Hasher& h,
                                typename Dataset::PointHandle p,
                                std::vector<typename Hasher::Value>& out ) { h.hash( p, out ); };

/// Hashes every point of `ds` with `h`, in parallel, and calls
/// `write_row(i, values)` with the hash values of point `i`. Runs with the GIL
/// released, so it must be called with the GIL held, and `write_row` must
/// neither throw nor touch Python objects.
///
/// `max_threads` must be at most the number of threads the hasher was
/// constructed for: `CrossPolytope` and `LatticeLSH` index their scratch space,
/// sized with `omp_get_max_threads()` at construction, by
/// `omp_get_thread_num()`. The team is capped at both `max_threads` and the
/// current `omp_get_max_threads()`, so that a later change of the thread count
/// cannot overflow the scratch space.
///
/// The scratch space is also shared by concurrent calls on the same hasher
/// (from several Python threads), so hashers without a `const` `hash` are
/// serialized through `scratch_mutex`.
template <typename Hasher, typename Dataset, typename Writer>
void hash_rows( Hasher& h,
                const Dataset& ds,
                int max_threads,
                std::mutex& scratch_mutex,
                Writer write_row ) {
    using Value = typename Hasher::Value;
    // Everything that may throw happens before the parallel region.
    const size_t n = ds.size();
    std::vector<typename Dataset::PointHandle> handles;
    handles.reserve( n );
    for ( size_t i = 0; i < n; i++ ) {
        handles.push_back( ds[i] );
    }

    nb::gil_scoped_release release;
    std::unique_lock<std::mutex> lock( scratch_mutex, std::defer_lock );
    if constexpr ( !ConstHasher<Hasher, Dataset> ) {
        lock.lock();
    }
    const int threads = std::max( 1, std::min( max_threads, omp_get_max_threads() ) );

#pragma omp parallel num_threads( threads )
    {
        std::vector<Value> values;
#pragma omp for schedule( dynamic, 64 )
        for ( size_t i = 0; i < n; i++ ) {
            h.hash( handles[i], values );
            write_row( i, values );
        }
    }
}

/// Element types of the arrays returned for each value type: `Code` for the
/// per-concatenation codes and `Packed` for the packed keys, which take
/// `packed_width` elements per key.
template <typename Value>
struct ValueTraits;

template <uint8_t K>
struct ValueTraits<panna::BitwiseLshValue<K>> {
    static constexpr size_t k = K;
    using Code = uint8_t;
    using Packed = uint32_t;
    static constexpr size_t packed_width = 1;
    static constexpr bool packed_is_void = false;
};

template <typename Symbol, uint8_t K>
struct ValueTraits<panna::SymbolLshValue<Symbol, K>> {
    static constexpr size_t k = K;
    using Code = Symbol;
    using Packed = uint8_t;
    static constexpr size_t packed_width = K * sizeof( Symbol );
    static constexpr bool packed_is_void = true;
};

/// Writers: codes, bitwise. Unpacks the `K` bits, concatenation 0 first.
template <uint8_t K>
inline void write_codes( const panna::BitwiseLshValue<K>& v, uint8_t* out ) {
    const uint32_t bits = v.get_bits();
    for ( size_t i = 0; i < K; i++ ) {
        out[i] = ( bits >> ( K - 1 - i ) ) & 1;
    }
}

/// Writers: codes, symbolic. Copies the `K` symbols.
template <typename Symbol, uint8_t K>
inline void write_codes( const panna::SymbolLshValue<Symbol, K>& v, Symbol* out ) {
    std::copy( v.data(), v.data() + K, out );
}

/// Writers: packed, bitwise. The bits as a `uint32`.
template <uint8_t K>
inline void write_packed( const panna::BitwiseLshValue<K>& v, uint32_t* out ) {
    *out = v.get_bits();
}

/// Writers: packed, symbolic. The raw bytes of the `K` symbols.
template <typename Symbol, uint8_t K>
inline void write_packed( const panna::SymbolLshValue<Symbol, K>& v, uint8_t* out ) {
    std::memcpy( out, v.data(), K * sizeof( Symbol ) );
}

/// Hashes `ds` with `h`, which outputs `repetitions` values per point, and
/// returns a NumPy array:
/// - `packed=false`: the codes, of shape `(n, repetitions, K)`;
/// - `packed=true`: one key per repetition, of shape `(n, repetitions)`, with
///   dtype `uint32` for bitwise values and `V{K*sizeof(Symbol)}` for the others.
/// With `single=true` (`ds` holds the single point of a 1-D input) the leading
/// `n` axis is dropped.
template <typename Hasher, typename Dataset>
nb::object hash_dataset( Hasher& h,
                         const Dataset& ds,
                         size_t repetitions,
                         bool packed,
                         bool single,
                         int max_threads,
                         std::mutex& scratch_mutex ) {
    using Value = typename Hasher::Value;
    using Traits = ValueTraits<Value>;
    const size_t n = ds.size();
    std::vector<size_t> shape;
    if ( !single ) {
        shape.push_back( n );
    }
    shape.push_back( repetitions );

    if ( !packed ) {
        using Code = typename Traits::Code;
        shape.push_back( Traits::k );
        auto [out, arr] = make_owned_array<Code>( shape );
        const size_t row_stride = repetitions * Traits::k;
        hash_rows( h, ds, max_threads, scratch_mutex,
                   [out = out, row_stride]( size_t i, const std::vector<Value>& values ) {
                       Code* row = out + i * row_stride;
                       for ( size_t r = 0; r < values.size(); r++ ) {
                           write_codes( values[r], row + r * Traits::k );
                       }
                   } );
        return nb::cast( arr );
    }

    using Packed = typename Traits::Packed;
    constexpr size_t width = Traits::packed_width;
    if constexpr ( Traits::packed_is_void ) {
        shape.push_back( width );
    }
    auto [out, arr] = make_owned_array<Packed>( shape );
    const size_t row_stride = repetitions * width;
    hash_rows( h, ds, max_threads, scratch_mutex,
               [out = out, row_stride]( size_t i, const std::vector<Value>& values ) {
                   Packed* row = out + i * row_stride;
                   for ( size_t r = 0; r < values.size(); r++ ) {
                       write_packed( values[r], row + r * width );
                   }
               } );
    nb::object result = nb::cast( arr );
    if constexpr ( Traits::packed_is_void ) {
        // View the trailing `width` bytes as one fixed-width void scalar, and
        // drop the resulting axis of length 1.
        nb::module_ np = nb::module_::import_( "numpy" );
        std::string dtype = "V" + std::to_string( width );
        result = result.attr( "view" )( np.attr( "dtype" )( dtype ) );
        shape.pop_back();
        nb::list new_shape;
        for ( size_t s : shape ) {
            new_shape.append( s );
        }
        result = result.attr( "reshape" )( nb::tuple( new_shape ) );
    }
    return result;
}

/// Hashes the points of a 1-D or 2-D float array with the hasher held by
/// `variant`, after copying them into a `Dataset` (see `to_dataset`).
template <typename Dataset, typename Variant>
nb::object hash_dense( Variant& variant,
                       const FloatArray& data,
                       size_t dimensions,
                       size_t repetitions,
                       bool packed,
                       int max_threads,
                       std::mutex& scratch_mutex ) {
    Dataset dataset = to_dataset<Dataset>( data, dimensions );
    const bool single = data.ndim() == 1;
    return std::visit(
        [&]( auto& h ) {
            return hash_dataset( h, dataset, repetitions, packed, single, max_threads,
                                 scratch_mutex );
        },
        variant );
}

/// Hashes the sets of a CSR structure (see `from_csr`) with the hasher held by
/// `variant`.
template <typename Variant>
nb::object hash_csr( Variant& variant,
                     const IndexArray& indptr,
                     const IndexArray& indices,
                     size_t repetitions,
                     bool packed,
                     int max_threads,
                     std::mutex& scratch_mutex ) {
    panna::SparseSets sets = from_csr( indptr, indices );
    return std::visit(
        [&]( auto& h ) {
            return hash_dataset( h, sets, repetitions, packed, false, max_threads,
                                 scratch_mutex );
        },
        variant );
}

/// `collision_probability` of the hasher held by `variant`, on a scalar.
template <typename Variant>
float collision_probability_scalar( Variant& variant, float distance ) {
    return std::visit( [distance]( auto& h ) { return h.collision_probability( distance ); },
                       variant );
}

/// `collision_probability` of the hasher held by `variant`, element-wise on an
/// array. Returns a `float32` array of the same shape.
template <typename Variant>
nb::ndarray<nb::numpy, float> collision_probability_array( Variant& variant,
                                                           const FloatArray& distances ) {
    std::vector<size_t> shape( distances.ndim() );
    for ( size_t i = 0; i < distances.ndim(); i++ ) {
        shape[i] = distances.shape( i );
    }
    auto [out, arr] = make_owned_array<float>( shape );
    const float* in = distances.data();
    const size_t size = distances.size();
    std::visit(
        [&]( auto& h ) {
            for ( size_t i = 0; i < size; i++ ) {
                out[i] = h.collision_probability( in[i] );
            }
        },
        variant );
    return arr;
}

/// The number of OpenMP threads the calling thread would use now. The family
/// wrappers record it when they build their hasher, and pass it as the
/// `max_threads` cap of `hash_rows`.
inline int current_max_threads() {
    return std::max( 1, omp_get_max_threads() );
}

// ---------------------------------------------------------------------------
// LSH family classes
// ---------------------------------------------------------------------------

template <uint8_t K>
using SimhashHasher = panna::Simhash<K, panna::UnitNormPoints, panna::CosineDistance>;

struct PySimhash {
    using Ks = KList<1, 2, 4, 8, 16, 32>;
    using Variant = HasherVariant<SimhashHasher, Ks>::type;

    size_t dimensions;
    size_t repetitions;
    size_t k;
    int max_threads;
    std::mutex scratch_mutex;
    Variant inner;

    PySimhash( size_t dimensions, size_t repetitions, size_t k, std::optional<uint64_t> seed ):
        dimensions( dimensions ),
        repetitions( repetitions ),
        k( k ),
        max_threads( current_max_threads() ),
        inner( build( dimensions, repetitions, k, seed ) ) {
    }

    static Variant
    build( size_t dimensions, size_t repetitions, size_t k, std::optional<uint64_t> seed ) {
        check_positive( dimensions, "dimensions" );
        check_positive( repetitions, "repetitions" );
        return dispatch_k<Ks>( k, [&]( auto kc ) {
            return with_global_rng( seed, [&] {
                return Variant( std::in_place_type<SimhashHasher<decltype( kc )::value>>,
                                dimensions, repetitions );
            } );
        } );
    }

    nb::object hash( const FloatArray& data, bool packed ) {
        return hash_dense<panna::UnitNormPoints>(
            inner, data, dimensions, repetitions, packed, max_threads, scratch_mutex );
    }

    std::string repr() const {
        std::stringstream sstream;
        sstream << "Simhash(dimensions=" << dimensions << ", k=" << k
                << ", repetitions=" << repetitions << ")";
        return sstream.str();
    }
};

template <uint8_t K>
using CrossPolytopeHasher = panna::CrossPolytope<K, panna::UnitNormPoints, panna::CosineDistance>;

struct PyCrossPolytope {
    using Ks = KList<1, 2, 4, 8, 16>;
    using Variant = HasherVariant<CrossPolytopeHasher, Ks>::type;
    /// The codes, in `[0, 2 * dimensions)`, are stored as `int16`.
    static constexpr size_t MAX_DIMENSIONS = 16384;

    size_t dimensions;
    size_t repetitions;
    size_t k;
    size_t estimation_repetitions;
    float estimation_eps;
    int max_threads;
    std::mutex scratch_mutex;
    Variant inner;

    PyCrossPolytope( size_t dimensions,
                     size_t repetitions,
                     size_t k,
                     size_t estimation_repetitions,
                     float estimation_eps,
                     std::optional<uint64_t> seed ):
        dimensions( dimensions ),
        repetitions( repetitions ),
        k( k ),
        estimation_repetitions( estimation_repetitions ),
        estimation_eps( estimation_eps ),
        max_threads( current_max_threads() ),
        inner( build( dimensions, repetitions, k, estimation_repetitions, estimation_eps, seed ) ) {
    }

    static Variant build( size_t dimensions,
                          size_t repetitions,
                          size_t k,
                          size_t estimation_repetitions,
                          float estimation_eps,
                          std::optional<uint64_t> seed ) {
        check_positive( dimensions, "dimensions" );
        if ( dimensions > MAX_DIMENSIONS ) {
            std::stringstream sstream;
            sstream << "dimensions must be at most " << MAX_DIMENSIONS;
            throw nb::value_error( sstream.str().c_str() );
        }
        check_positive( repetitions, "repetitions" );
        check_positive( estimation_repetitions, "estimation_repetitions" );
        if ( !( estimation_eps > 0.0f && estimation_eps <= 1.0f ) ) {
            throw nb::value_error( "estimation_eps must be in (0, 1]" );
        }
        return dispatch_k<Ks>( k, [&]( auto kc ) {
            return with_global_rng( seed, [&] {
                return Variant( std::in_place_type<CrossPolytopeHasher<decltype( kc )::value>>,
                                dimensions, repetitions, estimation_repetitions, estimation_eps );
            } );
        } );
    }

    nb::object hash( const FloatArray& data, bool packed ) {
        return hash_dense<panna::UnitNormPoints>(
            inner, data, dimensions, repetitions, packed, max_threads, scratch_mutex );
    }

    std::string repr() const {
        std::stringstream sstream;
        sstream << "CrossPolytope(dimensions=" << dimensions << ", k=" << k
                << ", repetitions=" << repetitions
                << ", estimation_repetitions=" << estimation_repetitions
                << ", estimation_eps=" << float_repr( estimation_eps ) << ")";
        return sstream.str();
    }
};

template <uint8_t K>
using E2LSHHasher = panna::E2LSH<K, panna::EuclideanPoints, panna::EuclideanDistance>;

/// Copies the points of a 2-D array to fit a hasher on. Raises `ValueError`
/// unless the array is 2-D and has at least `min_points` rows.
inline panna::EuclideanPoints fit_dataset( const FloatArray& data, size_t min_points ) {
    if ( data.ndim() != 2 ) {
        std::stringstream sstream;
        sstream << "fit expects a two-dimensional array, got " << data.ndim() << " dimensions";
        throw nb::value_error( sstream.str().c_str() );
    }
    if ( data.shape( 0 ) < min_points ) {
        std::stringstream sstream;
        if ( data.shape( 0 ) == 0 ) {
            sstream << "cannot fit on an empty dataset";
        } else {
            sstream << "fit needs at least " << min_points << " points, got " << data.shape( 0 );
        }
        throw nb::value_error( sstream.str().c_str() );
    }
    return to_dataset<panna::EuclideanPoints>( data );
}

struct PyE2LSH {
    using Ks = KList<1, 2, 4, 8, 16>;
    using Variant = HasherVariant<E2LSHHasher, Ks>::type;

    size_t dimensions;
    size_t repetitions;
    size_t k;
    int max_threads;
    std::mutex scratch_mutex;
    Variant inner;

    PyE2LSH( size_t dimensions,
             size_t repetitions,
             float quantization_width,
             size_t k,
             std::optional<uint64_t> seed ):
        dimensions( dimensions ),
        repetitions( repetitions ),
        k( k ),
        max_threads( current_max_threads() ),
        inner( build( dimensions, repetitions, quantization_width, k, seed ) ) {
    }

    PyE2LSH( size_t dimensions, size_t repetitions, size_t k, int max_threads, Variant&& inner ):
        dimensions( dimensions ),
        repetitions( repetitions ),
        k( k ),
        max_threads( max_threads ),
        inner( std::move( inner ) ) {
    }

    static Variant build( size_t dimensions,
                          size_t repetitions,
                          float quantization_width,
                          size_t k,
                          std::optional<uint64_t> seed ) {
        check_positive( dimensions, "dimensions" );
        check_positive( repetitions, "repetitions" );
        check_positive( quantization_width, "quantization_width" );
        return dispatch_k<Ks>( k, [&]( auto kc ) {
            return with_global_rng( seed, [&] {
                return Variant( std::in_place_type<E2LSHHasher<decltype( kc )::value>>,
                                quantization_width, dimensions, repetitions );
            } );
        } );
    }

    static std::unique_ptr<PyE2LSH> fit( const FloatArray& data,
                                         size_t repetitions,
                                         size_t k,
                                         float delta,
                                         std::optional<uint64_t> seed ) {
        check_positive( repetitions, "repetitions" );
        check_delta( delta );
        panna::EuclideanPoints points = fit_dataset( data, 1 );
        const size_t dimensions = points.get_dimensions();
        const int max_threads = current_max_threads();
        Variant inner = dispatch_k<Ks>( k, [&]( auto kc ) {
            constexpr uint8_t K = decltype( kc )::value;
            return with_global_rng( seed, [&] {
                panna::E2LSHBuilder<K, panna::EuclideanPoints, panna::EuclideanDistance> builder(
                    dimensions );
                builder.fit( points, repetitions, delta );
                return Variant( builder.build( repetitions ) );
            } );
        } );
        const float width = std::visit( []( auto& h ) { return h.get_quantization_width(); }, inner );
        if ( !( width > 0.0f ) || !std::isfinite( width ) ) {
            throw nb::value_error( "could not fit a positive quantization width: are all the "
                                   "projections of the data equal (for example, all-zero data)?" );
        }
        return std::make_unique<PyE2LSH>( dimensions, repetitions, k, max_threads,
                                          std::move( inner ) );
    }

    float quantization_width() const {
        return std::visit( []( auto& h ) { return h.get_quantization_width(); }, inner );
    }

    nb::object hash( const FloatArray& data, bool packed ) {
        return hash_dense<panna::EuclideanPoints>(
            inner, data, dimensions, repetitions, packed, max_threads, scratch_mutex );
    }

    std::string repr() const {
        std::stringstream sstream;
        sstream << "E2LSH(dimensions=" << dimensions << ", k=" << k
                << ", repetitions=" << repetitions
                << ", quantization_width=" << float_repr( quantization_width() ) << ")";
        return sstream.str();
    }
};

template <uint8_t K>
using LatticeHasher = panna::LatticeLSH<K, panna::EuclideanPoints, panna::EuclideanDistance>;

struct PyLatticeLSH {
    using Ks = KList<1, 2, 4, 8, 16>;
    using Variant = HasherVariant<LatticeHasher, Ks>::type;

    size_t dimensions;
    size_t repetitions;
    size_t k;
    int max_threads;
    std::mutex scratch_mutex;
    Variant inner;

    PyLatticeLSH( size_t dimensions,
                  size_t repetitions,
                  float scaling_factor,
                  size_t k,
                  const std::optional<FloatArray>& offset,
                  std::optional<uint64_t> seed ):
        dimensions( dimensions ),
        repetitions( repetitions ),
        k( k ),
        max_threads( current_max_threads() ),
        inner( build( dimensions, repetitions, scaling_factor, k, offset, seed ) ) {
    }

    PyLatticeLSH( size_t dimensions, size_t repetitions, size_t k, int max_threads, Variant&& inner ):
        dimensions( dimensions ),
        repetitions( repetitions ),
        k( k ),
        max_threads( max_threads ),
        inner( std::move( inner ) ) {
    }

    static Variant build( size_t dimensions,
                          size_t repetitions,
                          float scaling_factor,
                          size_t k,
                          const std::optional<FloatArray>& offset,
                          std::optional<uint64_t> seed ) {
        check_positive( dimensions, "dimensions" );
        check_positive( repetitions, "repetitions" );
        check_positive( scaling_factor, "scaling_factor" );
        std::vector<float> offset_vec( dimensions, 0.0f );
        if ( offset.has_value() ) {
            if ( offset->ndim() != 1 || offset->shape( 0 ) != dimensions ) {
                std::stringstream sstream;
                sstream << "offset must be a one-dimensional array of length " << dimensions;
                throw nb::value_error( sstream.str().c_str() );
            }
            std::copy( offset->data(), offset->data() + dimensions, offset_vec.begin() );
        }
        return dispatch_k<Ks>( k, [&]( auto kc ) {
            return with_global_rng( seed, [&] {
                return Variant( std::in_place_type<LatticeHasher<decltype( kc )::value>>,
                                offset_vec, scaling_factor, dimensions, repetitions );
            } );
        } );
    }

    static std::unique_ptr<PyLatticeLSH> fit( const FloatArray& data,
                                              size_t repetitions,
                                              size_t k,
                                              float delta,
                                              std::optional<uint64_t> seed ) {
        check_positive( repetitions, "repetitions" );
        check_delta( delta );
        // The fit computes spanning trees of the data, which need two points.
        panna::EuclideanPoints points = fit_dataset( data, 2 );
        const size_t dimensions = points.get_dimensions();
        // `LatticeLSH` sizes its per-thread scratch space with the thread
        // count of the calling thread, the same as `hash_rows` caps at.
        const int max_threads = current_max_threads();
        Variant inner = dispatch_k<Ks>( k, [&]( auto kc ) {
            constexpr uint8_t K = decltype( kc )::value;
            return with_global_rng( seed, [&] {
                panna::LatticeLSHBuilder<K, panna::EuclideanPoints, panna::EuclideanDistance>
                    builder( dimensions );
                builder.fit( points, repetitions, delta );
                return Variant( builder.build( repetitions ) );
            } );
        } );
        return std::make_unique<PyLatticeLSH>( dimensions, repetitions, k, max_threads,
                                               std::move( inner ) );
    }

    float scaling_factor() const {
        return std::visit( []( auto& h ) { return h.get_scaling_factor(); }, inner );
    }

    nb::ndarray<nb::numpy, float> offset() const {
        const std::vector<float>& offset =
            std::visit( []( auto& h ) -> const std::vector<float>& { return h.get_offset(); },
                        inner );
        auto [out, arr] = make_owned_array<float>( { offset.size() } );
        std::copy( offset.begin(), offset.end(), out );
        return arr;
    }

    nb::object hash( const FloatArray& data, bool packed ) {
        return hash_dense<panna::EuclideanPoints>(
            inner, data, dimensions, repetitions, packed, max_threads, scratch_mutex );
    }

    std::string repr() const {
        std::stringstream sstream;
        sstream << "LatticeLSH(dimensions=" << dimensions << ", k=" << k
                << ", repetitions=" << repetitions
                << ", scaling_factor=" << float_repr( scaling_factor() ) << ")";
        return sstream.str();
    }
};

template <uint8_t K>
using MinHashHasher = panna::Minhash<K, panna::SparseSets>;

template <uint8_t K>
using MinHash1BitHasher = panna::Minhash1Bit<K, panna::SparseSets>;

inline constexpr char MINHASH_NAME[] = "MinHash";
inline constexpr char MINHASH1BIT_NAME[] = "MinHash1Bit";

/// The Jaccard families, hashing sets given in CSR form.
template <template <uint8_t> class Hasher, typename KsT, const char* Name>
struct PyJaccardFamily {
    using Ks = KsT;
    using Variant = typename HasherVariant<Hasher, Ks>::type;

    size_t repetitions;
    size_t k;
    int max_threads;
    std::mutex scratch_mutex;
    Variant inner;

    PyJaccardFamily( size_t repetitions, size_t k, std::optional<uint64_t> seed ):
        repetitions( repetitions ),
        k( k ),
        max_threads( current_max_threads() ),
        inner( build( repetitions, k, seed ) ) {
    }

    static Variant build( size_t repetitions, size_t k, std::optional<uint64_t> seed ) {
        check_positive( repetitions, "repetitions" );
        return dispatch_k<Ks>( k, [&]( auto kc ) {
            return with_global_rng( seed, [&] {
                return Variant( std::in_place_type<Hasher<decltype( kc )::value>>, repetitions );
            } );
        } );
    }

    nb::object hash( const IndexArray& indptr, const IndexArray& indices, bool packed ) {
        return hash_csr( inner, indptr, indices, repetitions, packed, max_threads, scratch_mutex );
    }

    std::string repr() const {
        std::stringstream sstream;
        sstream << Name << "(k=" << k << ", repetitions=" << repetitions << ")";
        return sstream.str();
    }
};

using PyMinHash = PyJaccardFamily<MinHashHasher, KList<1, 2, 4, 8, 16>, MINHASH_NAME>;
using PyMinHash1Bit =
    PyJaccardFamily<MinHash1BitHasher, KList<1, 2, 4, 8, 16, 32>, MINHASH1BIT_NAME>;

// Signatures for the type stubs (`nb::sig`). nanobind would render the input
// arrays as `ndarray[dtype=float32, order='C', ...]` and the result of `hash`,
// which depends on `packed`, as `object`. The array inputs are rendered as
// `NDArray[Any]` rather than `ArrayLike`: nanobind converts the dtype and the
// layout of arrays implicitly, but it does not accept lists.
#define PANNA_NDARRAY_ANY "numpy.typing.NDArray[typing.Any]"
#define PANNA_NDARRAY_INT "numpy.typing.NDArray[numpy.integer]"

/// The `nb::sig` of a `hash` method taking `params` (after `self`) and
/// returning `NDArray[numpy.<codes>]`, or `NDArray[numpy.<packed>]` when
/// `packed` is true. Besides the signature proper (the last line, used at
/// runtime by `help()` and by error messages), it holds two `@overload`
/// declarations on `packed: Literal[False]` and `Literal[True]`, which
/// `stubgen` copies verbatim into the stub. nanobind simplifies the type
/// names after the first parenthesis only, so the first decorator is spelled
/// `overload`: its import comes from the `typing.overload` of the others.
std::string hash_sig( const std::string& params, const char* codes, const char* packed ) {
    const std::string codes_t = std::string( "numpy.typing.NDArray[numpy." ) + codes + "]";
    const std::string packed_t = std::string( "numpy.typing.NDArray[numpy." ) + packed + "]";
    const std::string head = "def hash(self, " + params + ", packed: ";
    return "@overload\n" + head + "typing.Literal[False] = False) -> " + codes_t + ": ...\n" +
           "@typing.overload\n" + head + "typing.Literal[True]) -> " + packed_t + ": ...\n" +
           "@typing.overload\n" + head + "bool = False) -> " + codes_t + " | " + packed_t;
}

/// Registers the methods shared by every family class: the two overloads of
/// `collision_probability` (documented by `cp_doc`), the `k` and
/// `repetitions` properties, `__repr__` and `__str__`.
template <typename Py>
void def_common( nb::class_<Py>& cls, const char* cp_doc, const char* k_doc ) {
    // The scalar overload takes a handle so that it can refuse NumPy arrays:
    // nanobind's converting `float` caster would otherwise accept a size-1
    // array of another dtype (through the deprecated `ndarray.__float__`) and
    // return a float. NumPy scalars (`np.float64(0.5)`) still give a float.
    cls.def(
           "collision_probability",
           []( Py& self, nb::handle distance ) {
               float value;
               if ( nb::isinstance( distance, nb::module_::import_( "numpy" ).attr( "ndarray" ) ) ||
                    !nb::try_cast<float>( distance, value ) ) {
                   throw nb::next_overload();
               }
               return collision_probability_scalar( self.inner, value );
           },
           nb::arg( "distance" ),
           nb::sig( "def collision_probability(self, distance: float) -> float" ),
           cp_doc )
        .def(
            "collision_probability",
            []( Py& self, const FloatArray& distances ) {
                return collision_probability_array( self.inner, distances );
            },
            nb::arg( "distance" ),
            nb::sig( "def collision_probability(self, distance: " PANNA_NDARRAY_ANY
                     ") -> numpy.typing.NDArray[numpy.float32]" ) )
        .def_prop_ro( "k", []( const Py& self ) { return self.k; }, k_doc )
        .def_prop_ro( "repetitions", []( const Py& self ) { return self.repetitions; },
                      "Number of independent hash functions." )
        .def( "__repr__", &Py::repr )
        .def( "__str__", &Py::repr );
}

/// Registers the `hash` method and the `dimensions` property of the families
/// hashing dense float vectors. `codes` and `packed` name the NumPy scalar
/// types of the result of `hash` (see `hash_sig`).
template <typename Py>
void def_dense( nb::class_<Py>& cls, const char* codes, const char* packed, const char* hash_doc ) {
    const std::string sig = hash_sig( "data: " PANNA_NDARRAY_ANY, codes, packed );
    cls.def( "hash",
             &Py::hash,
             nb::arg( "data" ),
             nb::arg( "packed" ) = false,
             nb::sig( sig.c_str() ),
             hash_doc )
        .def_prop_ro( "dimensions", []( const Py& self ) { return self.dimensions; },
                      "Dimensionality of the input points." );
}

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
        size_t refine_iterations = 10;

        if (kwargs.contains("repetitions")) {
            repetitions = nb::cast<size_t>(kwargs["repetitions"]);
        }
        if (kwargs.contains("delta")) {
            delta = nb::cast<double>(kwargs["delta"]);
        }
        if (kwargs.contains("epsilon")) {
            epsilon = nb::cast<float>(kwargs["epsilon"]);
        }
        if (kwargs.contains("refine_iterations")) {
            refine_iterations = nb::cast<size_t>(kwargs["refine_iterations"]);
        }
        LOG_INFO("refine_iterations", refine_iterations);

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
                    dimensions, repetitions, data_cpp, delta, epsilon, refine_iterations );
            break;
        case E2LSH:
            inner
                .emplace<panna::EMST<panna::EuclideanPoints, E2LSHHasher, panna::EuclideanDistance>>(
                    dimensions, repetitions, data_cpp, delta, epsilon, refine_iterations );
            break;
        case CrossPolytope:
            inner.emplace<CrossPolytopeEMST>(
                dimensions, repetitions, data_cpp, delta, epsilon, refine_iterations );
            break;
        case Simhash:
            inner.emplace<SimhashEMST>(
                dimensions, repetitions, data_cpp, delta, epsilon, refine_iterations );
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
        // TODO: move this reweighting to emst.hpp
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
            (*tree_vec_ptr)[i * 3 + 0] = tree_edges_vec[i].a;
            (*tree_vec_ptr)[i * 3 + 1] = tree_edges_vec[i].b;
            (*tree_vec_ptr)[i * 3 + 2] = tree_edges_vec[i].weight;
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

        // FIXME: watch out that these are not Euclidean if the user chose
        // simhash or cross polytope
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
        // fast_hdbscan builds this array with a KD-tree self-query of
        // k+1 columns, where each point appears among its own neighbors
        // (at distance 0). CoreDistances instead stores exactly k
        // neighbors excluding the point itself, so we widen each row to
        // k+1 and pad with the point's own index to match the expected
        // shape and content. Note two remaining differences from
        // fast_hdbscan: self is in the last column rather than the
        // first, and rows are in heap order, not sorted by distance.
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

/// Runs `panna::pair_forest_emst` on `data` with the given hash family.
///
/// The cosine-based families search the tree under the cosine distance on
/// unit-normalized points; as in `EMST_exposed::reweight_with_euclidean`, the
/// edges are reported with their Euclidean distance instead.
template <typename Dataset, typename Hasher, typename Distance>
nb::tuple run_pair_forest_emst( const nb::ndarray<float, nb::c_contig>& data_in,
                                float epsilon,
                                float delta,
                                size_t repetitions ) {
    Dataset dataset = to_dataset<Dataset>( data_in );

    panna::PairEmstResult res =
        panna::pair_forest_emst<Dataset, Hasher, Distance>( dataset, epsilon, delta, repetitions );

    if constexpr ( std::is_same_v<Dataset, panna::UnitNormPoints> ) {
        for ( auto& e : res.tree ) {
            e.weight = panna::EuclideanDistance::compute( dataset[e.a], dataset[e.b] );
        }
    }

    nb::list profile;
    for (auto el : res.profile) {
        nb::dict elpy;
        elpy["elapsed_ms"] = el.elapsed_ms;
        elpy["prefix"] = el.prefix;
        elpy["repetition"] = el.repetition;
        elpy["emst_confirmed_weight"] = el.emst_confirmed_weight;
        elpy["emst_weight_lower_bound"] = el.emst_weight_lower_bound;
        elpy["emst_max_weight"] = el.emst_max_weight;
        elpy["emst_total_weight"] = el.emst_total_weight;
        elpy["emst_num_confirmed"] = el.emst_num_confirmed;
        profile.append(elpy);
    }

    nb::dict stats;
    stats["distance_count"] = res.distances_computed;
    stats["index_size_bytes"] = res.index_bytes;
    stats["prefix_at_stop"] = res.prefix_at_stop;
    stats["repetitions_at_stop"] = res.repetitions_at_stop;
    stats["seed_ms"] = res.seed_ms;
    stats["index_ms"] = res.index_ms;
    stats["discovery_ms"] = res.discovery_ms;
    stats["profile"] = profile;

    nb::tuple tree = tree_to_pytuple( res.tree );
    return nb::make_tuple( tree[0], tree[1], stats );
}

nb::tuple pair_forest_emst( const nb::ndarray<float, nb::c_contig>& data_in, nb::kwargs kwargs ) {
    if ( data_in.ndim() != 2 ) {
        throw nb::value_error( "pair_forest_emst: expected a two-dimensional array" );
    }

    size_t repetitions = 512;
    float delta = 0.1;
    float epsilon = 0.2;
    EuclideanHashFamily family = EuclideanHashFamily::Lattice;

    if ( kwargs.contains( "repetitions" ) ) {
        repetitions = nb::cast<size_t>( kwargs["repetitions"] );
    }
    if ( kwargs.contains( "delta" ) ) {
        delta = nb::cast<float>( kwargs["delta"] );
    }
    if ( kwargs.contains( "epsilon" ) ) {
        epsilon = nb::cast<float>( kwargs["epsilon"] );
    }
    if ( kwargs.contains( "family" ) ) {
        family = string_to_family( nb::cast<std::string>( kwargs["family"] ) );
    }

    switch ( family ) {
    case Lattice:
        return run_pair_forest_emst<panna::EuclideanPoints,
                                    EMST_exposed::LatticeHasher,
                                    panna::EuclideanDistance>(
            data_in, epsilon, delta, repetitions );
    case E2LSH:
        return run_pair_forest_emst<panna::EuclideanPoints,
                                    EMST_exposed::E2LSHHasher,
                                    panna::EuclideanDistance>(
            data_in, epsilon, delta, repetitions );
    case CrossPolytope:
        return run_pair_forest_emst<panna::UnitNormPoints,
                                    EMST_exposed::CrossPolytopeHasher,
                                    panna::CosineDistance>( data_in, epsilon, delta, repetitions );
    case Simhash:
        return run_pair_forest_emst<panna::UnitNormPoints,
                                    EMST_exposed::SimhashHasher,
                                    panna::CosineDistance>( data_in, epsilon, delta, repetitions );
    }
    throw std::logic_error( "unreachable" );
}

/// Runs `panna::pair_forest_emst_mutual_reachability` on `data` with the given
/// hash family.
///
/// The tree weights are already Euclidean mutual-reachability distances; the
/// core distances are converted from `Distance` units to Euclidean ones, so
/// that both are on the same scale for every family.
template <typename Dataset, typename Hasher, typename Distance>
nb::tuple run_pair_forest_emst_mutual_reachability( const nb::ndarray<float, nb::c_contig>& data_in,
                                                    size_t num_neighbors,
                                                    float epsilon,
                                                    float delta,
                                                    size_t repetitions,
                                                    size_t refine_iterations ) {
    Dataset dataset = to_dataset<Dataset>( data_in );

    panna::PairMrEmstResult res =
        panna::pair_forest_emst_mutual_reachability<Dataset, Hasher, Distance>(
            dataset, num_neighbors, epsilon, delta, repetitions, refine_iterations );

    /// With no neighbors `CoreDistances` stores nothing, and every core
    /// distance is zero.
    const size_t num_points = res.core_distances.size();
    float* cores = new float[num_points];
    for ( size_t i = 0; i < num_points; i++ ) {
        cores[i] = res.core_distances.get_num_neighbors() == 0
                       ? 0.0f
                       : Distance::to_euclidean( res.core_distances.core_distance( i ) );
    }
    nb::capsule cores_owner( cores, []( void* p ) noexcept { delete[] (float*)p; } );
    const auto cores_numpy =
        nb::ndarray<nb::numpy, float, nb::ndim<1>>( cores, { num_points }, cores_owner );

    nb::dict stats;
    stats["distance_count"] = res.distances_computed;
    stats["index_size_bytes"] = res.index_bytes;
    stats["prefix_at_stop"] = res.prefix_at_stop;
    stats["repetitions_at_stop"] = res.repetitions_at_stop;
    stats["seed_ms"] = res.seed_ms;
    stats["index_ms"] = res.index_ms;
    stats["discovery_ms"] = res.discovery_ms;

    nb::tuple tree = tree_to_pytuple( res.tree );
    return nb::make_tuple( tree[0], tree[1], cores_numpy, stats );
}

nb::tuple pair_forest_emst_mutual_reachability( const nb::ndarray<float, nb::c_contig>& data_in,
                                                nb::kwargs kwargs ) {
    if ( data_in.ndim() != 2 ) {
        throw nb::value_error(
            "pair_forest_emst_mutual_reachability: expected a two-dimensional array" );
    }

    size_t num_neighbors = 5;
    size_t repetitions = 512;
    float delta = 0.1;
    float epsilon = 0.2;
    size_t refine_iterations = 10;
    EuclideanHashFamily family = EuclideanHashFamily::Lattice;

    if ( kwargs.contains( "num_neighbors" ) ) {
        num_neighbors = nb::cast<size_t>( kwargs["num_neighbors"] );
    }
    if ( kwargs.contains( "repetitions" ) ) {
        repetitions = nb::cast<size_t>( kwargs["repetitions"] );
    }
    if ( kwargs.contains( "delta" ) ) {
        delta = nb::cast<float>( kwargs["delta"] );
    }
    if ( kwargs.contains( "epsilon" ) ) {
        epsilon = nb::cast<float>( kwargs["epsilon"] );
    }
    if ( kwargs.contains( "refine_iterations" ) ) {
        refine_iterations = nb::cast<size_t>( kwargs["refine_iterations"] );
    }
    if ( kwargs.contains( "family" ) ) {
        family = string_to_family( nb::cast<std::string>( kwargs["family"] ) );
    }

    switch ( family ) {
    case Lattice:
        return run_pair_forest_emst_mutual_reachability<panna::EuclideanPoints,
                                                        EMST_exposed::LatticeHasher,
                                                        panna::EuclideanDistance>(
            data_in, num_neighbors, epsilon, delta, repetitions, refine_iterations );
    case E2LSH:
        return run_pair_forest_emst_mutual_reachability<panna::EuclideanPoints,
                                                        EMST_exposed::E2LSHHasher,
                                                        panna::EuclideanDistance>(
            data_in, num_neighbors, epsilon, delta, repetitions, refine_iterations );
    case CrossPolytope:
        return run_pair_forest_emst_mutual_reachability<panna::UnitNormPoints,
                                                        EMST_exposed::CrossPolytopeHasher,
                                                        panna::CosineDistance>(
            data_in, num_neighbors, epsilon, delta, repetitions, refine_iterations );
    case Simhash:
        return run_pair_forest_emst_mutual_reachability<panna::UnitNormPoints,
                                                        EMST_exposed::SimhashHasher,
                                                        panna::CosineDistance>(
            data_in, num_neighbors, epsilon, delta, repetitions, refine_iterations );
    }
    throw std::logic_error( "unreachable" );
}

nb::tuple emst_theory_of_computing( nb::ndarray<float, nb::c_contig>& data_in, nb::kwargs kwargs ) {
    float delta = 0.1;
    float gamma = 1.0;

    if ( kwargs.contains( "delta" ) ) {
        delta = nb::cast<float>( kwargs["delta"] );
    }
    if ( kwargs.contains( "gamma" ) ) {
        gamma = nb::cast<float>( kwargs["gamma"] );
    }

    panna::NormedPoints dataset = to_dataset<panna::NormedPoints>( data_in );
    size_t nrows = dataset.size();
    size_t dimensionality = dataset.get_dimensions();
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
    panna::NormedPoints dataset = to_dataset<panna::NormedPoints>( data_in );

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
    panna::NormedPoints dataset = to_dataset<panna::NormedPoints>( data_in );

    float diam = 
        panna::approximate_diameter<panna::EuclideanDistance>( dataset );
    return diam;
}

NB_MODULE( _panna_impl, m ) {
    m.attr( "git_version" ) = GIT_COMMIT_HASH;
    m.def(
        "set_seed",
        []( uint64_t seed ) {
            // Released GIL, then the lock: see `with_global_rng`.
            nb::gil_scoped_release release;
            std::lock_guard<std::mutex> lock( global_rng_mutex );
            panna::seed_global_rng( seed );
        },
        nb::arg( "seed" ),
        "Set the seed of the global random number generator used by the panna module." );
    m.def( "emst_theory_of_computing",
           &emst_theory_of_computing ),
    m.def( "distance_histogram",
           &distance_histogram ),
    m.def( "approximate_diameter",
           &approximate_diameter ),
    m.attr( "pair_forest_emst_version" ) = panna::PAIR_EMST_VERSION;
    m.def( "pair_forest_emst",
           &pair_forest_emst,
           "Approximate Euclidean minimum spanning tree of a NumPy array of data\n"
           "points, via the OpenMP-driven PairForestIndex search.\n\n"
           "Returns a tuple (weights, edges, stats): the edge weights, the (n-1, 2)\n"
           "array of edge endpoints, and a dict of execution statistics.\n\n"
           "Keyword arguments:\n"
           "  repetitions (int, default 512): repetitions of the LSH index.\n"
           "  delta (float, default 0.1): probability of failure.\n"
           "  epsilon (float, default 0.2): approximation factor.\n"
           "  family (str, default 'lattice'): hash family to use." ),
    m.def( "pair_forest_emst_mutual_reachability",
           &pair_forest_emst_mutual_reachability,
           "Approximate minimum spanning tree of a NumPy array of data points under\n"
           "the mutual reachability distance, via the PairForestIndex search.\n\n"
           "Returns a tuple (weights, edges, core_distances, stats): the Euclidean\n"
           "mutual-reachability weights of the edges, the (n-1, 2) array of edge\n"
           "endpoints, the (n,) array of Euclidean core distances, and a dict of\n"
           "execution statistics.\n\n"
           "Keyword arguments:\n"
           "  num_neighbors (int, default 5): neighbors defining the core distances.\n"
           "  repetitions (int, default 512): repetitions of the LSH index.\n"
           "  delta (float, default 0.1): probability of failure.\n"
           "  epsilon (float, default 0.2): approximation factor.\n"
           "  refine_iterations (int, default 10): rounds of NN-descent used to\n"
           "    sharpen the core distances, 0 to disable.\n"
           "  family (str, default 'lattice'): hash family to use." ),
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
             "Constructs the EMST index from a NumPy array of data points.\n\n"
             "Keyword arguments:\n"
             "  repetitions (int, default 500): repetitions of the LSH index.\n"
             "  delta (float, default 0.1): probability of failure.\n"
             "  epsilon (float, default 0.2): approximation factor.\n"
             "  refine_iterations (int, default 10): rounds of NN-descent used to\n"
             "    sharpen the seeded core distances, 0 to disable. Only affects\n"
             "    find_mst_dbscan.\n"
             "  family (str, default 'lattice'): hash family to use.")
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

    // Docstring fragments shared by the LSH family classes.
#define PANNA_SEED_DOC                                                                    \
    "  seed (int, optional): when given, re-seeds the module-wide global\n"              \
    "    random number generator (the one of `set_seed`) before sampling.\n"              \
    "    This affects all later random draws of the module. Sampling holds a\n"           \
    "    module-wide lock, so a seeded hasher is reproducible even when other\n"          \
    "    Python threads build hashers concurrently.\n"
#define PANNA_PARALLEL_DOC "Rows are hashed in parallel with OpenMP, with the GIL released."
#define PANNA_DENSE_INPUT_DOC                                                             \
    "  data (ndarray): a (n, dimensions) array of points, or a single\n"                \
    "    (dimensions,) point. Converted to float32 if needed.\n"
#define PANNA_DENSE_RAISES_DOC                                                            \
    "Raises:\n"                                                                           \
    "  ValueError: if the array is not 1-D or 2-D, or its last axis does\n"              \
    "    not match `dimensions`."
#define PANNA_VOID_KEYS_DOC                                                               \
    "  The void keys support equality, `np.unique`, and use as dict keys via\n"           \
    "  `.tobytes()`; their ordering does not match the order of the codes.\n"
#define PANNA_CP_RETURNS_DOC                                                              \
    "Returns:\n"                                                                          \
    "  A float for a float input, or a float32 array of the same shape for an\n"          \
    "  array input."

    nb::class_<PySimhash> simhash(
        m,
        "Simhash",
        "SimHash (random hyperplane) LSH family for the cosine distance.\n\n"
        "Each of the `repetitions` independent hash functions concatenates `k`\n"
        "random hyperplanes: concatenation `i` is 1 if the point lies on the\n"
        "positive side of hyperplane `i`, and 0 otherwise. Input points are\n"
        "normalized to unit length (and stored in 16-bit fixed point)\n"
        "internally, hence scaling a point does not change its hash.\n\n"
        PANNA_PARALLEL_DOC );
    simhash.def( nb::init<size_t, size_t, size_t, std::optional<uint64_t>>(),
                 nb::arg( "dimensions" ),
                 nb::arg( "repetitions" ),
                 nb::arg( "k" ) = 16,
                 nb::arg( "seed" ) = nb::none(),
                 "Samples a new Simhash hasher.\n\n"
                 "Args:\n"
                 "  dimensions (int): dimensionality of the input points.\n"
                 "  repetitions (int): number of independent hash functions.\n"
                 "  k (int, default 16): number of concatenated hyperplanes per hash\n"
                 "    function; one of 1, 2, 4, 8, 16, 32.\n"
                 PANNA_SEED_DOC "\n"
                 "Raises:\n"
                 "  ValueError: if `k` is not supported, or `dimensions` or\n"
                 "    `repetitions` is zero." );
    def_dense( simhash,
               "uint8",
               "uint32",
               "Hashes the given points.\n\n"
               "Args:\n" PANNA_DENSE_INPUT_DOC
               "  packed (bool, default False): return one key per repetition\n"
               "    rather than the individual bits.\n\n"
               "Returns:\n"
               "  If `packed` is False, a uint8 array of shape (n, repetitions, k)\n"
               "  holding the bits (0 or 1) of each concatenation. If `packed` is\n"
               "  True, a uint32 array of shape (n, repetitions), where concatenation\n"
               "  `i` is bit `k - 1 - i` of the key. A 1-D input drops the leading `n`\n"
               "  axis.\n\n" PANNA_DENSE_RAISES_DOC );
    def_common( simhash,
                "Probability that a single concatenation (one hyperplane) collides\n"
                "on two points at the given cosine distance `1 - cos(angle)`, that\n"
                "is `1 - angle / pi`. A whole repetition collides with probability\n"
                "`collision_probability(d) ** k`.\n\n"
                "Args:\n"
                "  distance (float or ndarray): cosine distance(s), in [0, 2]; values\n"
                "    outside this range give NaN.\n\n" PANNA_CP_RETURNS_DOC,
                "Number of concatenated hyperplanes per hash function." );

    nb::class_<PyCrossPolytope> crosspolytope(
        m,
        "CrossPolytope",
        "Cross-polytope LSH family for the cosine distance.\n\n"
        "Each of the `repetitions` independent hash functions concatenates `k`\n"
        "cross-polytope hashes: concatenation `i` applies a pseudo-random\n"
        "rotation to the point and returns the index of the closest signed axis,\n"
        "an integer in [0, 2 * dimensions). Input points are normalized to unit\n"
        "length (and stored in 16-bit fixed point) internally, hence scaling a\n"
        "point does not change its hash.\n\n"
        "The collision probability has no closed form: it is estimated by\n"
        "Monte Carlo when the hasher is built (see `estimation_repetitions` and\n"
        "`estimation_eps`), with a fixed internal seed.\n\n"
        PANNA_PARALLEL_DOC );
    crosspolytope.def( nb::init<size_t, size_t, size_t, size_t, float, std::optional<uint64_t>>(),
                       nb::arg( "dimensions" ),
                       nb::arg( "repetitions" ),
                       nb::arg( "k" ) = 4,
                       nb::arg( "estimation_repetitions" ) = 1024,
                       nb::arg( "estimation_eps" ) = 5e-3,
                       nb::arg( "seed" ) = nb::none(),
                       "Samples a new CrossPolytope hasher.\n\n"
                       "Args:\n"
                       "  dimensions (int): dimensionality of the input points, at most\n"
                       "    16384.\n"
                       "  repetitions (int): number of independent hash functions.\n"
                       "  k (int, default 4): number of concatenated cross-polytope hashes\n"
                       "    per hash function; one of 1, 2, 4, 8, 16.\n"
                       "  estimation_repetitions (int, default 1024): number of Monte Carlo\n"
                       "    trials per segment of the collision probability table.\n"
                       "  estimation_eps (float, default 0.005): width, in dot product, of\n"
                       "    the segments of the collision probability table; in (0, 1].\n"
                       "    The table has 2 / estimation_eps segments.\n"
                       PANNA_SEED_DOC "\n"
                       "Raises:\n"
                       "  ValueError: if `k` is not supported, `dimensions` is zero or\n"
                       "    larger than 16384, `repetitions` or `estimation_repetitions`\n"
                       "    is zero, or `estimation_eps` is out of range." );
    def_dense( crosspolytope,
               "int16",
               "void",
               "Hashes the given points.\n\n"
               "Args:\n" PANNA_DENSE_INPUT_DOC
               "  packed (bool, default False): return one key per repetition\n"
               "    rather than the individual codes.\n\n"
               "Returns:\n"
               "  If `packed` is False, an int16 array of shape (n, repetitions, k)\n"
               "  holding the code, in [0, 2 * dimensions), of each concatenation. If\n"
               "  `packed` is True, an array of shape (n, repetitions) with void dtype\n"
               "  `V{2*k}`, holding the raw bytes of the `k` codes of each repetition.\n"
               PANNA_VOID_KEYS_DOC
               "  A 1-D input drops the leading `n` axis.\n\n" PANNA_DENSE_RAISES_DOC );
    def_common( crosspolytope,
                "Probability that a single concatenation collides on two points at the\n"
                "given cosine distance `1 - cos(angle)`, looked up in the table\n"
                "estimated at construction. A whole repetition collides with\n"
                "probability `collision_probability(d) ** k`.\n\n"
                "Args:\n"
                "  distance (float or ndarray): cosine distance(s), in [0, 2]; values\n"
                "    outside this range are clamped.\n\n" PANNA_CP_RETURNS_DOC,
                "Number of concatenated cross-polytope hashes per hash function." );
    crosspolytope
        .def_prop_ro( "estimation_repetitions",
                      []( const PyCrossPolytope& self ) { return self.estimation_repetitions; },
                      "Number of Monte Carlo trials per segment of the collision\n"
                      "probability table." )
        .def_prop_ro( "estimation_eps",
                      []( const PyCrossPolytope& self ) { return self.estimation_eps; },
                      "Width, in dot product, of the segments of the collision\n"
                      "probability table." );

    nb::class_<PyE2LSH> e2lsh(
        m,
        "E2LSH",
        "E2LSH (p-stable, Datar et al.) LSH family for the Euclidean distance.\n\n"
        "Each of the `repetitions` independent hash functions concatenates `k`\n"
        "quantized random projections: concatenation `i` is\n"
        "`floor((dot(a_i, x) + b_i) / w)`, with `a_i` a standard Gaussian vector,\n"
        "`b_i` uniform in [0, w), and `w` the quantization width.\n\n"
        "The quantization width is either given explicitly to the constructor,\n"
        "or fitted on data with the static method `fit`.\n\n"
        PANNA_PARALLEL_DOC );
    e2lsh.def( nb::init<size_t, size_t, float, size_t, std::optional<uint64_t>>(),
               nb::arg( "dimensions" ),
               nb::arg( "repetitions" ),
               nb::arg( "quantization_width" ),
               nb::arg( "k" ) = 8,
               nb::arg( "seed" ) = nb::none(),
               "Samples a new E2LSH hasher with the given quantization width.\n\n"
               "Args:\n"
               "  dimensions (int): dimensionality of the input points.\n"
               "  repetitions (int): number of independent hash functions.\n"
               "  quantization_width (float): the width `w` of the quantization\n"
               "    buckets; positive.\n"
               "  k (int, default 8): number of concatenated projections per hash\n"
               "    function; one of 1, 2, 4, 8, 16.\n"
               PANNA_SEED_DOC "\n"
               "Raises:\n"
               "  ValueError: if `k` is not supported, `dimensions` or `repetitions`\n"
               "    is zero, or `quantization_width` is not positive." );
    e2lsh.def_static( "fit",
                      &PyE2LSH::fit,
                      nb::arg( "data" ),
                      nb::arg( "repetitions" ),
                      nb::arg( "k" ) = 8,
                      nb::arg( "delta" ) = 0.1,
                      nb::arg( "seed" ) = nb::none(),
                      nb::sig( "def fit(data: " PANNA_NDARRAY_ANY ", repetitions: int, k: int = 8, "
                               "delta: float = 0.1, seed: int | None = None) -> E2LSH" ),
                      "Fits the quantization width on data, and samples a new E2LSH\n"
                      "hasher with it.\n\n"
                      "The width is searched so that, on a sample of the data, the\n"
                      "average number of colliding pairs per repetition is about the\n"
                      "square root of the sample size. The fit runs with the GIL\n"
                      "released.\n\n"
                      "Args:\n"
                      "  data (ndarray): a (n, dimensions) array of points, with n >= 1.\n"
                      "    Converted to float32 if needed.\n"
                      "  repetitions (int): number of independent hash functions.\n"
                      "  k (int, default 8): number of concatenated projections per hash\n"
                      "    function; one of 1, 2, 4, 8, 16.\n"
                      "  delta (float, default 0.1): target failure probability, in\n"
                      "    (0, 1). The current fitting heuristic does not depend on it.\n"
                      PANNA_SEED_DOC "\n"
                      "Returns:\n"
                      "  A new E2LSH hasher.\n\n"
                      "Raises:\n"
                      "  ValueError: if `data` is not 2-D or is empty, `k` is not\n"
                      "    supported, `repetitions` is zero, `delta` is out of range, or\n"
                      "    no positive width can be fitted (for example, on all-zero\n"
                      "    data)." );
    def_dense( e2lsh,
               "int8",
               "void",
               "Hashes the given points.\n\n"
               "Args:\n" PANNA_DENSE_INPUT_DOC
               "  packed (bool, default False): return one key per repetition\n"
               "    rather than the individual codes.\n\n"
               "Returns:\n"
               "  If `packed` is False, an int8 array of shape (n, repetitions, k)\n"
               "  holding the quantized projection of each concatenation. The codes\n"
               "  are truncated to 8 bits, so they wrap around when the quantization\n"
               "  width is very small relative to the range of the projections. If\n"
               "  `packed` is True, an array of shape (n, repetitions) with void dtype\n"
               "  `V{k}`, holding the raw bytes of the `k` codes of each repetition.\n"
               PANNA_VOID_KEYS_DOC
               "  A 1-D input drops the leading `n` axis.\n\n" PANNA_DENSE_RAISES_DOC );
    def_common( e2lsh,
                "Probability that a single concatenation collides on two points at the\n"
                "given Euclidean distance (Datar et al., 2004). A whole repetition\n"
                "collides with probability `collision_probability(d) ** k`.\n\n"
                "Args:\n"
                "  distance (float or ndarray): Euclidean distance(s), non-negative.\n\n"
                PANNA_CP_RETURNS_DOC,
                "Number of concatenated projections per hash function." );
    e2lsh.def_prop_ro( "quantization_width",
                       &PyE2LSH::quantization_width,
                       "The width `w` of the quantization buckets." );

    nb::class_<PyLatticeLSH> lattice(
        m,
        "LatticeLSH",
        "E8-lattice LSH family for the Euclidean distance.\n\n"
        "Each of the `repetitions` independent hash functions concatenates `k`\n"
        "lattice hashes: concatenation `i` maps the point, translated by\n"
        "`-offset` and divided by `scaling_factor`, to 8 random dimensions, and\n"
        "returns an integer encoding the closest point of the randomly shifted\n"
        "E8 lattice.\n\n"
        "The offset and scaling factor are either given explicitly to the\n"
        "constructor, or fitted on data with the static method `fit`.\n\n"
        PANNA_PARALLEL_DOC );
    lattice.def( nb::init<size_t,
                          size_t,
                          float,
                          size_t,
                          const std::optional<FloatArray>&,
                          std::optional<uint64_t>>(),
                 nb::arg( "dimensions" ),
                 nb::arg( "repetitions" ),
                 nb::arg( "scaling_factor" ),
                 nb::arg( "k" ) = 4,
                 nb::arg( "offset" ) = nb::none(),
                 nb::arg( "seed" ) = nb::none(),
                 nb::sig( "def __init__(self, dimensions: int, repetitions: int, "
                          "scaling_factor: float, k: int = 4, "
                          "offset: " PANNA_NDARRAY_ANY " | None = None, "
                          "seed: int | None = None) -> None" ),
                 "Samples a new LatticeLSH hasher with the given parameters.\n\n"
                 "Args:\n"
                 "  dimensions (int): dimensionality of the input points.\n"
                 "  repetitions (int): number of independent hash functions.\n"
                 "  scaling_factor (float): the points are divided by this factor\n"
                 "    before hashing, so larger values give larger buckets; positive.\n"
                 "  k (int, default 4): number of concatenated lattice hashes per hash\n"
                 "    function; one of 1, 2, 4, 8, 16.\n"
                 "  offset (ndarray, optional): a (dimensions,) array subtracted from\n"
                 "    the points before hashing; zeros by default.\n"
                 PANNA_SEED_DOC "\n"
                 "Raises:\n"
                 "  ValueError: if `k` is not supported, `dimensions` or `repetitions`\n"
                 "    is zero, `scaling_factor` is not positive, or `offset` does not\n"
                 "    have shape (dimensions,)." );
    lattice.def_static( "fit",
                        &PyLatticeLSH::fit,
                        nb::arg( "data" ),
                        nb::arg( "repetitions" ),
                        nb::arg( "k" ) = 4,
                        nb::arg( "delta" ) = 0.1,
                        nb::arg( "seed" ) = nb::none(),
                        nb::sig( "def fit(data: " PANNA_NDARRAY_ANY ", repetitions: int, k: int = 4, "
                                 "delta: float = 0.1, seed: int | None = None) -> LatticeLSH" ),
                        "Fits the offset and scaling factor on data, and samples a new\n"
                        "LatticeLSH hasher with them.\n\n"
                        "The offset is the mean point of the data. The scaling factor is\n"
                        "the smallest one for which `repetitions` hash functions with a\n"
                        "single concatenation fail to make the endpoints of the heaviest\n"
                        "edge of an approximate minimum spanning tree of the data collide\n"
                        "with probability below `delta`. The fit runs with the GIL\n"
                        "released.\n\n"
                        "If the environment variable `PANNA_LATTICE_SCALING_FACTOR` is set\n"
                        "to a positive number, the scaling factor is not fitted and that\n"
                        "value is used instead (the offset is still the mean point).\n\n"
                        "Args:\n"
                        "  data (ndarray): a (n, dimensions) array of points, with n >= 2.\n"
                        "    Converted to float32 if needed.\n"
                        "  repetitions (int): number of independent hash functions.\n"
                        "  k (int, default 4): number of concatenated lattice hashes per\n"
                        "    hash function; one of 1, 2, 4, 8, 16.\n"
                        "  delta (float, default 0.1): target failure probability, in\n"
                        "    (0, 1).\n"
                        PANNA_SEED_DOC "\n"
                        "Returns:\n"
                        "  A new LatticeLSH hasher.\n\n"
                        "Raises:\n"
                        "  ValueError: if `data` is not 2-D or has fewer than 2 points, `k`\n"
                        "    is not supported, `repetitions` is zero, or `delta` is out of\n"
                        "    range." );
    def_dense( lattice,
               "int64",
               "void",
               "Hashes the given points.\n\n"
               "Args:\n" PANNA_DENSE_INPUT_DOC
               "  packed (bool, default False): return one key per repetition\n"
               "    rather than the individual codes.\n\n"
               "Returns:\n"
               "  If `packed` is False, an int64 array of shape (n, repetitions, k)\n"
               "  holding the encoded lattice point of each concatenation. If `packed`\n"
               "  is True, an array of shape (n, repetitions) with void dtype `V{8*k}`,\n"
               "  holding the raw bytes of the `k` codes of each repetition.\n"
               PANNA_VOID_KEYS_DOC
               "  A 1-D input drops the leading `n` axis.\n\n" PANNA_DENSE_RAISES_DOC );
    def_common( lattice,
                "Probability that a single concatenation collides on two points at the\n"
                "given Euclidean distance, looked up in a precomputed table of the E8\n"
                "lattice collision probabilities. A whole repetition collides with\n"
                "probability `collision_probability(d) ** k`.\n\n"
                "Args:\n"
                "  distance (float or ndarray): Euclidean distance(s), non-negative.\n\n"
                PANNA_CP_RETURNS_DOC,
                "Number of concatenated lattice hashes per hash function." );
    lattice
        .def_prop_ro( "scaling_factor",
                      &PyLatticeLSH::scaling_factor,
                      "The factor the points are divided by before hashing." )
        .def_prop_ro( "offset",
                      &PyLatticeLSH::offset,
                      // The array owns its data: the default `reference_internal`
                      // policy of properties does not apply.
                      nb::rv_policy::move,
                      nb::sig( "def offset(self) -> numpy.typing.NDArray[numpy.float32]" ),
                      "The (dimensions,) offset subtracted from the points before\n"
                      "hashing, as a new float32 array." );

#define PANNA_CSR_HASH_DOC( CODES )                                                       \
    "Hashes the sets of a CSR structure: set `i` holds the tokens\n"                       \
    "`indices[indptr[i]:indptr[i+1]]`. Repeated tokens do not change the\n"                \
    "hash, and the empty set hashes to token 0. See also `hash_sets`.\n\n"                \
    "Args:\n"                                                                             \
    "  indptr (ndarray): a (n + 1,) integer array, starting at 0,\n"                      \
    "    non-decreasing, and ending at len(indices).\n"                                    \
    "  indices (ndarray): an integer array of tokens, in [0, 2**32).\n"                    \
    "  packed (bool, default False): return one key per repetition\n"                     \
    "    rather than the individual codes.\n\n"                                           \
    "Returns:\n" CODES "\n"                                                               \
    "Raises:\n"                                                                           \
    "  ValueError: if `indptr` or `indices` is malformed."

    nb::class_<PyMinHash> minhash(
        m,
        "MinHash",
        "MinHash LSH family for the Jaccard distance between sets of integer\n"
        "tokens.\n\n"
        "Each of the `repetitions` independent hash functions concatenates `k`\n"
        "MinHash values: concatenation `i` is the token of the set with the\n"
        "smallest value under a random tabulation hash function.\n\n"
        PANNA_PARALLEL_DOC );
    minhash.def( nb::init<size_t, size_t, std::optional<uint64_t>>(),
                 nb::arg( "repetitions" ),
                 nb::arg( "k" ) = 4,
                 nb::arg( "seed" ) = nb::none(),
                 "Samples a new MinHash hasher.\n\n"
                 "Args:\n"
                 "  repetitions (int): number of independent hash functions.\n"
                 "  k (int, default 4): number of concatenated MinHash values per\n"
                 "    hash function; one of 1, 2, 4, 8, 16.\n"
                 PANNA_SEED_DOC "\n"
                 "Raises:\n"
                 "  ValueError: if `k` is not supported, or `repetitions` is zero." );
    minhash.def( "hash",
                 &PyMinHash::hash,
                 nb::arg( "indptr" ),
                 nb::arg( "indices" ),
                 nb::arg( "packed" ) = false,
                 nb::sig( hash_sig( "indptr: " PANNA_NDARRAY_INT ", indices: " PANNA_NDARRAY_INT,
                                   "int32",
                                   "void" )
                              .c_str() ),
                 PANNA_CSR_HASH_DOC(
                     "  If `packed` is False, an int32 array of shape (n, repetitions, k)\n"
                     "  holding the minimum token of each concatenation (tokens above\n"
                     "  2**31 appear as negative values). If `packed` is True, an array of\n"
                     "  shape (n, repetitions) with void dtype `V{4*k}`, holding the raw\n"
                     "  bytes of the `k` codes of each repetition.\n" PANNA_VOID_KEYS_DOC ) );
    def_common( minhash,
                "Probability that a single concatenation collides on two sets at the\n"
                "given Jaccard distance `d`, that is `1 - d`. A whole repetition\n"
                "collides with probability `collision_probability(d) ** k`.\n\n"
                "Args:\n"
                "  distance (float or ndarray): Jaccard distance(s), in [0, 1].\n\n"
                PANNA_CP_RETURNS_DOC,
                "Number of concatenated MinHash values per hash function." );

    nb::class_<PyMinHash1Bit> minhash1bit(
        m,
        "MinHash1Bit",
        "One-bit MinHash LSH family for the Jaccard distance between sets of\n"
        "integer tokens.\n\n"
        "Like `MinHash`, but each concatenation keeps only the lowest bit of the\n"
        "minimum token, so that a whole repetition fits in a uint32 key.\n\n"
        PANNA_PARALLEL_DOC );
    minhash1bit.def( nb::init<size_t, size_t, std::optional<uint64_t>>(),
                     nb::arg( "repetitions" ),
                     nb::arg( "k" ) = 16,
                     nb::arg( "seed" ) = nb::none(),
                     "Samples a new MinHash1Bit hasher.\n\n"
                     "Args:\n"
                     "  repetitions (int): number of independent hash functions.\n"
                     "  k (int, default 16): number of concatenated MinHash bits per\n"
                     "    hash function; one of 1, 2, 4, 8, 16, 32.\n"
                     PANNA_SEED_DOC "\n"
                     "Raises:\n"
                     "  ValueError: if `k` is not supported, or `repetitions` is zero." );
    minhash1bit.def( "hash",
                     &PyMinHash1Bit::hash,
                     nb::arg( "indptr" ),
                     nb::arg( "indices" ),
                     nb::arg( "packed" ) = false,
                     nb::sig( hash_sig( "indptr: " PANNA_NDARRAY_INT ", indices: " PANNA_NDARRAY_INT,
                                       "uint8",
                                       "uint32" )
                                  .c_str() ),
                     PANNA_CSR_HASH_DOC(
                         "  If `packed` is False, a uint8 array of shape (n, repetitions, k)\n"
                         "  holding the bits (0 or 1) of each concatenation. If `packed` is\n"
                         "  True, a uint32 array of shape (n, repetitions), where\n"
                         "  concatenation `i` is bit `k - 1 - i` of the key.\n" ) );
    def_common( minhash1bit,
                "Probability that a single concatenation collides on two sets at the\n"
                "given Jaccard distance `d`, that is `1 - d / 2`: the minimum tokens\n"
                "are equal with probability `1 - d`, and otherwise their lowest bits\n"
                "agree half of the time. A whole repetition collides with probability\n"
                "`collision_probability(d) ** k`.\n\n"
                "Args:\n"
                "  distance (float or ndarray): Jaccard distance(s), in [0, 1].\n\n"
                PANNA_CP_RETURNS_DOC,
                "Number of concatenated MinHash bits per hash function." );

#undef PANNA_SEED_DOC
#undef PANNA_PARALLEL_DOC
#undef PANNA_DENSE_INPUT_DOC
#undef PANNA_DENSE_RAISES_DOC
#undef PANNA_VOID_KEYS_DOC
#undef PANNA_CP_RETURNS_DOC
#undef PANNA_CSR_HASH_DOC
#undef PANNA_NDARRAY_ANY
#undef PANNA_NDARRAY_INT
}
