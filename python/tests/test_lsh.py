"""Unit tests for the LSH family bindings (`Simhash`, `CrossPolytope`, `E2LSH`,
`LatticeLSH`, `MinHash`, `MinHash1Bit`)."""

import numpy as np
import pytest

import pypanna as pp

D = 24  # dimensions of the dense test data


# ---------------------------------------------------------------------------
# Family descriptions
# ---------------------------------------------------------------------------


class Family:
    """Describes how to build and feed one LSH family in the tests."""

    def __init__(self, name, make, ks, codes_dtype, symbol_bytes, dense):
        self.name = name
        self._make = make
        self.ks = ks
        self.codes_dtype = np.dtype(codes_dtype)
        # None for the bitwise families (packed into uint32).
        self.symbol_bytes = symbol_bytes
        self.dense = dense

    def make(self, k, repetitions=8, seed=None):
        return self._make(k=k, repetitions=repetitions, seed=seed)

    def packed_dtype(self, k):
        if self.symbol_bytes is None:
            return np.dtype(np.uint32)
        return np.dtype(f"V{self.symbol_bytes * k}")

    def hash(self, h, data, packed=False):
        if self.dense:
            return h.hash(data, packed=packed)
        return h.hash_sets(data, packed=packed)

    def __repr__(self):
        return self.name


FAMILIES = [
    Family(
        "Simhash",
        lambda k, repetitions, seed: pp.Simhash(D, repetitions, k=k, seed=seed),
        [1, 2, 4, 8, 16, 32],
        np.uint8,
        None,
        True,
    ),
    Family(
        "CrossPolytope",
        lambda k, repetitions, seed: pp.CrossPolytope(
            D, repetitions, k=k, estimation_repetitions=64, seed=seed
        ),
        [1, 2, 4, 8, 16],
        np.int16,
        2,
        True,
    ),
    Family(
        "E2LSH",
        lambda k, repetitions, seed: pp.E2LSH(D, repetitions, 2.0, k=k, seed=seed),
        [1, 2, 4, 8, 16],
        np.int8,
        1,
        True,
    ),
    Family(
        "LatticeLSH",
        lambda k, repetitions, seed: pp.LatticeLSH(D, repetitions, 2.0, k=k, seed=seed),
        [1, 2, 4, 8, 16],
        np.int64,
        8,
        True,
    ),
    Family(
        "MinHash",
        lambda k, repetitions, seed: pp.MinHash(repetitions, k=k, seed=seed),
        [1, 2, 4, 8, 16],
        np.int32,
        4,
        False,
    ),
    Family(
        "MinHash1Bit",
        lambda k, repetitions, seed: pp.MinHash1Bit(repetitions, k=k, seed=seed),
        [1, 2, 4, 8, 16, 32],
        np.uint8,
        None,
        False,
    ),
]

DENSE_FAMILIES = [f for f in FAMILIES if f.dense]
SET_FAMILIES = [f for f in FAMILIES if not f.dense]

FAMILY_K = [pytest.param(f, k, id=f"{f.name}-k{k}") for f in FAMILIES for k in f.ks]


def _dense_data(n=200, seed=0):
    return np.random.default_rng(seed).normal(size=(n, D))


def _set_data(n=200, seed=0):
    rng = np.random.default_rng(seed)
    return [
        rng.choice(1000, size=rng.integers(1, 40), replace=False).tolist()
        for _ in range(n)
    ]


def _data(family, n=200, seed=0):
    return _dense_data(n, seed) if family.dense else _set_data(n, seed)


def _near_duplicates(family, n=200, seed=0):
    """Pairs of rows (row i, row n + i) that are close, so that some of their
    hash keys collide and some do not."""
    rng = np.random.default_rng(seed)
    if family.dense:
        x = rng.normal(size=(n, D))
        y = x + rng.normal(scale=0.15, size=(n, D))
        return np.concatenate([x, y])
    xs = _set_data(n, seed)
    ys = []
    for s in xs:
        t = list(s)
        t[0] = 5000 + int(rng.integers(1000))  # replace one token
        ys.append(t)
    return xs + ys


def _pack_bits(codes):
    k = codes.shape[-1]
    w = np.uint64(1) << np.arange(k - 1, -1, -1, dtype=np.uint64)
    return (codes.astype(np.uint64) * w).sum(-1).astype(np.uint32)


# ---------------------------------------------------------------------------
# Shapes, dtypes, packing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("family, k", FAMILY_K)
def test_shapes_and_dtypes(family, k):
    h = family.make(k, repetitions=8, seed=5)
    data = _data(family, n=50)
    codes = family.hash(h, data)
    keys = family.hash(h, data, packed=True)
    assert codes.shape == (50, 8, k)
    assert codes.dtype == family.codes_dtype
    assert keys.shape == (50, 8)
    assert keys.dtype == family.packed_dtype(k)
    # The packed keys are the codes laid out next to each other.
    if family.symbol_bytes is None:
        assert set(np.unique(codes)) <= {0, 1}
        np.testing.assert_array_equal(_pack_bits(codes), keys)
    else:
        view = np.ascontiguousarray(codes).view(family.packed_dtype(k))[..., 0]
        np.testing.assert_array_equal(view, keys)


@pytest.mark.parametrize("family", DENSE_FAMILIES, ids=str)
def test_one_dimensional_input(family):
    h = family.make(4, repetitions=8, seed=1)
    X = _dense_data(10)
    codes = h.hash(X)
    keys = h.hash(X, packed=True)
    single = h.hash(X[3])
    single_keys = h.hash(X[3], packed=True)
    assert single.shape == (8, 4)
    assert single_keys.shape == (8,)
    np.testing.assert_array_equal(single, codes[3])
    np.testing.assert_array_equal(single_keys, keys[3])


@pytest.mark.parametrize("family", FAMILIES, ids=str)
def test_unsupported_k(family):
    with pytest.raises(ValueError) as excinfo:
        family.make(3)
    message = str(excinfo.value)
    for k in family.ks:
        assert str(k) in message
    with pytest.raises(ValueError):
        family.make(2 * family.ks[-1])


# ---------------------------------------------------------------------------
# Determinism and seeding
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("family", FAMILIES, ids=str)
def test_same_seed_same_outputs(family):
    data = _data(family)
    a = family.hash(family.make(4, seed=11), data)
    b = family.hash(family.make(4, seed=11), data)
    np.testing.assert_array_equal(a, b)
    assert repr(family.make(4, seed=11)) == repr(family.make(4, seed=11))


@pytest.mark.parametrize("family", FAMILIES, ids=str)
def test_different_seed_different_outputs(family):
    data = _data(family)
    a = family.hash(family.make(4, seed=11), data)
    b = family.hash(family.make(4, seed=12), data)
    assert not np.array_equal(a, b)


# ---------------------------------------------------------------------------
# Packed keys vs codes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("family", FAMILIES, ids=str)
@pytest.mark.parametrize("k", [2, 4])
def test_packed_equal_iff_codes_equal(family, k):
    n = 300
    h = family.make(k, repetitions=16, seed=3)
    data = _near_duplicates(family, n=n)
    codes = family.hash(h, data)
    keys = family.hash(h, data, packed=True)
    eq_keys = keys[:n] == keys[n:]
    eq_codes = (codes[:n] == codes[n:]).all(-1)
    # Both branches must be exercised for the test to be meaningful.
    assert eq_codes.any() and not eq_codes.all()
    np.testing.assert_array_equal(eq_keys, eq_codes)
    # And also on unrelated pairs.
    eq_keys = keys[: n // 2] == keys[n // 2 : n]
    eq_codes = (codes[: n // 2] == codes[n // 2 : n]).all(-1)
    np.testing.assert_array_equal(eq_keys, eq_codes)


# ---------------------------------------------------------------------------
# Parallel vs sequential
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("family", FAMILIES, ids=str)
def test_parallel_equals_sequential(family):
    h = family.make(family.ks[-1], repetitions=16, seed=7)
    data = _data(family, n=300)
    codes = family.hash(h, data)
    keys = family.hash(h, data, packed=True)
    seq = np.stack([family.hash(h, data[i : i + 1])[0] for i in range(len(data))])
    seq_keys = np.stack(
        [family.hash(h, data[i : i + 1], packed=True)[0] for i in range(len(data))]
    )
    np.testing.assert_array_equal(seq, codes)
    np.testing.assert_array_equal(seq_keys, keys)


@pytest.mark.parametrize(
    "make",
    [
        pytest.param(lambda: pp.CrossPolytope(D, 32, k=4, estimation_repetitions=64, seed=1), id="CrossPolytope"),
        pytest.param(lambda: pp.LatticeLSH(D, 32, 2.0, k=4, seed=1), id="LatticeLSH"),
    ],
)
def test_scratch_built_with_one_thread(make):
    """Regression: `CrossPolytope` and `LatticeLSH` size their per-thread
    scratch space at construction; hashing with more threads than at
    construction must neither crash nor change the result."""
    threadpoolctl = pytest.importorskip("threadpoolctl")
    X = _dense_data(2000)
    with threadpoolctl.threadpool_limits(limits=1, user_api="openmp"):
        h = make()
        expected = h.hash(X)
    with threadpoolctl.threadpool_limits(limits=4, user_api="openmp"):
        np.testing.assert_array_equal(h.hash(X), expected)
    np.testing.assert_array_equal(h.hash(X), expected)


# ---------------------------------------------------------------------------
# Collision probabilities
# ---------------------------------------------------------------------------

R = 4000  # repetitions for the collision rate estimates


def _pair_at(dist, kind, rng):
    """Two points at the given cosine or Euclidean distance."""
    x = rng.normal(size=D)
    x /= np.linalg.norm(x)
    u = rng.normal(size=D)
    u -= u.dot(x) * x
    u /= np.linalg.norm(u)
    if kind == "cosine":
        cos = 1 - dist
        return x, cos * x + np.sqrt(max(0.0, 1 - cos * cos)) * u
    return 3 * x, 3 * x + dist * u


def _assert_rate(emp, p, n, slack, table_samples=None):
    """Checks an empirical collision rate over `n` independent hash functions
    against the predicted probability `p`, within 5 standard deviations. When
    `p` is itself a Monte Carlo estimate from `table_samples` samples, its
    variance is added."""
    var = max(p * (1 - p), 1e-4)
    var = var / n + (var / table_samples if table_samples else 0.0)
    assert abs(emp - p) <= 5 * np.sqrt(var) + slack, f"empirical {emp:.4f}, predicted {p:.4f}"


# CrossPolytope tabulates Monte Carlo estimates of the collision probability at
# the dot products -1 + i * eps, and looks up the entry at or below the queried
# dot product. A coarse grid keeps the construction cheap; querying just above
# grid points makes the lookup unbiased.
CP_EPS = 0.05
CP_TABLE_SAMPLES = 1024
CP_DISTANCES = [1 - (-1 + CP_EPS * i + 1e-3) for i in (38, 30, 20)]  # dotp ~0.9, 0.5, 0


@pytest.mark.parametrize(
    "make, kind, dists, slack, table_samples",
    [
        # Simhash's and E2LSH's probabilities are exact.
        pytest.param(lambda: pp.Simhash(D, R, k=1, seed=3), "cosine",
                     [1 - np.cos(a) for a in (0.3, np.pi / 3, np.pi / 2, 2.0)], 0.0, None,
                     id="Simhash"),
        pytest.param(lambda: pp.CrossPolytope(D, R, k=1, estimation_repetitions=CP_TABLE_SAMPLES,
                                              estimation_eps=CP_EPS, seed=3),
                     "cosine", CP_DISTANCES, 0.005, CP_TABLE_SAMPLES, id="CrossPolytope"),
        pytest.param(lambda: pp.E2LSH(D, R, 2.0, k=1, seed=3), "euclidean",
                     [0.5, 1.5, 4.0], 0.0, None, id="E2LSH"),
        # LatticeLSH looks up a precomputed table on a distance grid.
        pytest.param(lambda: pp.LatticeLSH(D, R, 2.0, k=1, seed=3), "euclidean",
                     [0.5, 1.5, 4.0], 0.02, None, id="LatticeLSH"),
    ],
)
def test_collision_probability_dense(make, kind, dists, slack, table_samples):
    rng = np.random.default_rng(42)
    h = make()
    for dist in dists:
        x, y = _pair_at(dist, kind, rng)
        kx, ky = h.hash(np.stack([x, y]), packed=True)
        _assert_rate(float(np.mean(kx == ky)), h.collision_probability(dist), R, slack,
                     table_samples)


def test_simhash_collision_probability_is_angular():
    h = pp.Simhash(D, 4, k=1, seed=0)
    for angle in [0.0, 0.5, np.pi / 2, 2.5]:
        assert h.collision_probability(1 - np.cos(angle)) == pytest.approx(
            1 - angle / np.pi, abs=1e-3
        )


@pytest.mark.parametrize("family", SET_FAMILIES, ids=str)
def test_collision_probability_sets(family):
    h = family.make(1, repetitions=R, seed=9)
    n_union = 200
    for jd in [0.2, 0.5, 0.8]:
        n_inter = int(round((1 - jd) * n_union))
        common = np.arange(n_inter)
        rest = np.arange(n_inter, n_union) + 10_000
        half = len(rest) // 2
        a = np.concatenate([common, rest[:half]])
        b = np.concatenate([common, rest[half:]])
        true_jd = 1 - n_inter / n_union
        ka, kb = h.hash_sets([a, b], packed=True)
        _assert_rate(float(np.mean(ka == kb)), h.collision_probability(true_jd), R, 0.0)


@pytest.mark.parametrize("family", FAMILIES, ids=str)
def test_collision_probability_scalar_and_array(family):
    h = family.make(4, seed=1)
    p = h.collision_probability(0.5)
    assert isinstance(p, float)
    assert 0.0 <= p <= 1.0
    for arr in [np.array([0.1, 0.5]), np.array([[0.1, 0.5], [0.3, 0.2]], dtype=np.float32)]:
        out = h.collision_probability(arr)
        assert isinstance(out, np.ndarray)
        assert out.shape == arr.shape
        for idx in np.ndindex(arr.shape):
            assert out[idx] == pytest.approx(h.collision_probability(float(arr[idx])), abs=1e-6)
    # Regression: a size-1 float64 array used to be converted to a float.
    single = h.collision_probability(np.array([0.5]))
    assert isinstance(single, np.ndarray) and single.shape == (1,)
    # Python ints and NumPy scalars are scalars.
    for scalar in [1, np.float64(0.5), np.float32(0.5)]:
        assert isinstance(h.collision_probability(scalar), float)
    with pytest.raises(TypeError):
        h.collision_probability([0.5])


# ---------------------------------------------------------------------------
# Fitting (E2LSH, LatticeLSH)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def fit_data():
    return np.random.default_rng(1).normal(size=(3000, 8)).astype(np.float32)


def test_e2lsh_fit(fit_data):
    h = pp.E2LSH.fit(fit_data, repetitions=16, seed=1)
    assert h.quantization_width > 0
    assert (h.k, h.dimensions, h.repetitions) == (8, 8, 16)
    h2 = pp.E2LSH.fit(fit_data, repetitions=16, seed=1)
    assert h.quantization_width == h2.quantization_width
    np.testing.assert_array_equal(h.hash(fit_data), h2.hash(fit_data))


def test_lattice_fit(fit_data):
    h = pp.LatticeLSH.fit(fit_data, repetitions=16, k=2, delta=0.2, seed=1)
    assert h.scaling_factor > 0
    assert (h.k, h.dimensions, h.repetitions) == (2, 8, 16)
    assert h.offset.shape == (8,) and h.offset.dtype == np.float32
    np.testing.assert_allclose(h.offset, fit_data.mean(0), atol=1e-4)
    h2 = pp.LatticeLSH.fit(fit_data, repetitions=16, k=2, delta=0.2, seed=1)
    assert h.scaling_factor == h2.scaling_factor
    np.testing.assert_array_equal(h.hash(fit_data), h2.hash(fit_data))


def test_lattice_fit_large_then_small(fit_data):
    """Regression: fitting on a small dataset after a large one used to sample
    out-of-range indices."""
    pp.LatticeLSH.fit(fit_data, repetitions=4)
    assert pp.LatticeLSH.fit(fit_data[:5], repetitions=4).scaling_factor > 0
    assert pp.LatticeLSH.fit(fit_data[:2], repetitions=4).scaling_factor > 0


def test_e2lsh_fit_single_point(fit_data):
    assert pp.E2LSH.fit(fit_data[:1], repetitions=4).quantization_width > 0


@pytest.mark.parametrize("cls", [pp.E2LSH, pp.LatticeLSH], ids=lambda c: c.__name__)
def test_fit_invalid(cls, fit_data):
    with pytest.raises(ValueError):
        cls.fit(np.zeros((0, 8), dtype=np.float32), repetitions=4)
    with pytest.raises(ValueError):
        cls.fit(np.zeros(8, dtype=np.float32), repetitions=4)
    with pytest.raises(ValueError):
        cls.fit(fit_data, repetitions=4, k=5)
    with pytest.raises(ValueError):
        cls.fit(fit_data, repetitions=4, delta=1.5)


def test_fit_degenerate():
    with pytest.raises(ValueError):
        pp.LatticeLSH.fit(np.ones((1, 8), dtype=np.float32), repetitions=4)
    with pytest.raises(ValueError):
        pp.E2LSH.fit(np.zeros((10, 4), dtype=np.float32), repetitions=4)


def test_lattice_scaling_factor_env_override(fit_data, monkeypatch):
    monkeypatch.setenv("PANNA_LATTICE_SCALING_FACTOR", "3.5")
    assert pp.LatticeLSH.fit(fit_data, repetitions=4).scaling_factor == 3.5
    monkeypatch.delenv("PANNA_LATTICE_SCALING_FACTOR")
    assert pp.LatticeLSH.fit(fit_data, repetitions=4).scaling_factor != 3.5


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("family", DENSE_FAMILIES, ids=str)
def test_dimension_mismatch(family):
    h = family.make(4)
    with pytest.raises(ValueError):
        h.hash(np.zeros((3, D + 1)))
    with pytest.raises(ValueError):
        h.hash(np.zeros(D - 1))
    with pytest.raises(ValueError):
        h.hash(np.zeros((2, 3, D)))


@pytest.mark.parametrize(
    "build",
    [
        pytest.param(lambda: pp.E2LSH(4, 4, 0.0), id="e2lsh-width-zero"),
        pytest.param(lambda: pp.E2LSH(4, 4, -1.0), id="e2lsh-width-negative"),
        pytest.param(lambda: pp.E2LSH(4, 4, float("nan")), id="e2lsh-width-nan"),
        pytest.param(lambda: pp.LatticeLSH(4, 4, 0.0), id="lattice-scaling-zero"),
        pytest.param(lambda: pp.LatticeLSH(4, 4, -1.0), id="lattice-scaling-negative"),
        pytest.param(lambda: pp.LatticeLSH(4, 4, 1.0, offset=np.zeros(3)), id="lattice-offset-length"),
        pytest.param(lambda: pp.LatticeLSH(4, 4, 1.0, offset=np.zeros((2, 2))), id="lattice-offset-2d"),
        pytest.param(lambda: pp.CrossPolytope(4, 4, estimation_eps=0.0), id="cp-eps-zero"),
        pytest.param(lambda: pp.CrossPolytope(4, 4, estimation_repetitions=0), id="cp-est-reps-zero"),
        pytest.param(lambda: pp.MinHash(0), id="minhash-zero-reps"),
    ],
)
def test_invalid_constructor_arguments(build):
    with pytest.raises(ValueError):
        build()


@pytest.mark.parametrize("family", SET_FAMILIES, ids=str)
def test_malformed_csr(family):
    h = family.make(4)
    with pytest.raises(ValueError):  # non-monotonic indptr
        h.hash(np.array([0, 2, 1]), np.array([1, 2]))
    with pytest.raises(ValueError):  # indptr[-1] != len(indices)
        h.hash(np.array([0, 3]), np.array([1, 2]))
    with pytest.raises(ValueError):  # negative token
        h.hash_sets([[1, -2]])
    with pytest.raises(ValueError):  # set that is not one-dimensional
        h.hash_sets([[[1, 2]]])
    with pytest.raises(TypeError):  # non-integer token
        h.hash_sets([[1.5]])


def test_lattice_offset_is_a_translation():
    rng = np.random.default_rng(4)
    offset = np.array([1, 2, 3, 4])
    plain = pp.LatticeLSH(4, 8, 1.0, seed=1)
    shifted = pp.LatticeLSH(4, 8, 1.0, offset=offset, seed=1)
    np.testing.assert_array_equal(shifted.offset, offset)
    Y = rng.normal(size=(50, 4))
    np.testing.assert_array_equal(plain.hash(Y), shifted.hash(Y + offset))


# ---------------------------------------------------------------------------
# Cosine normalization
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("family", [f for f in FAMILIES if f.name in ("Simhash", "CrossPolytope")], ids=str)
def test_cosine_scale_invariance(family):
    h = family.make(4, repetitions=16, seed=1)
    X = _dense_data(100)
    expected = h.hash(X)
    for scale in [3.7, 0.01, 1000.0]:
        np.testing.assert_array_equal(h.hash(X * scale), expected)


# ---------------------------------------------------------------------------
# hash_sets
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("family", SET_FAMILIES, ids=str)
def test_hash_sets_inputs(family):
    h = family.make(4, repetitions=16, seed=2)
    sets = _set_data(100)
    codes = h.hash_sets(sets)
    indptr = np.concatenate([[0], np.cumsum([len(s) for s in sets])])
    indices = np.concatenate([np.asarray(s, dtype=np.int64) for s in sets])
    np.testing.assert_array_equal(h.hash(indptr, indices), codes)
    np.testing.assert_array_equal(
        h.hash(indptr.astype(np.int32), indices.astype(np.uint32)), codes
    )
    np.testing.assert_array_equal(h.hash_sets([set(s) for s in sets]), codes)
    np.testing.assert_array_equal(h.hash_sets([tuple(s) for s in sets]), codes)
    np.testing.assert_array_equal(h.hash_sets(iter([np.array(s) for s in sets])), codes)
    np.testing.assert_array_equal(
        h.hash_sets(sets, packed=True), h.hash(indptr, indices, packed=True)
    )
    assert h.hash_sets([]).shape == (0, 16, 4)
    # A set's hash does not depend on the order or multiplicity of its tokens.
    np.testing.assert_array_equal(
        h.hash_sets([sets[0][::-1] + sets[0]])[0], codes[0]
    )


@pytest.mark.parametrize("family", SET_FAMILIES, ids=str)
def test_hash_sets_scipy(family):
    sp = pytest.importorskip("scipy.sparse")
    h = family.make(4, repetitions=16, seed=2)
    sets = _set_data(100)
    codes = h.hash_sets(sets)
    rows = np.repeat(np.arange(len(sets)), [len(s) for s in sets])
    cols = np.concatenate(sets)
    M = sp.csr_matrix((np.ones(len(cols)), (rows, cols)), shape=(len(sets), 1000))
    np.testing.assert_array_equal(h.hash_sets(M), codes)
    np.testing.assert_array_equal(h.hash_sets(M.tocoo()), codes)
    np.testing.assert_array_equal(h.hash_sets(sp.csr_array(M)), codes)


@pytest.mark.parametrize("family", SET_FAMILIES, ids=str)
def test_empty_set_hashes_like_token_zero(family):
    # Every minhash of the empty set is token 0, i.e. the same codes as the
    # set {0}: all zeros for MinHash, and all-zero bits (0 % 2) for MinHash1Bit.
    h = family.make(4, repetitions=16, seed=2)
    empty, zero = h.hash_sets([[], [0]])
    np.testing.assert_array_equal(empty, zero)
    assert (empty == 0).all()
    empty_key, zero_key = h.hash_sets([[], [0]], packed=True)
    assert empty_key.tobytes() == zero_key.tobytes()


# ---------------------------------------------------------------------------
# Properties and repr
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("family", FAMILIES, ids=str)
def test_properties_and_repr(family):
    h = family.make(4, repetitions=8, seed=1)
    assert h.k == 4
    assert h.repetitions == 8
    if family.dense:
        assert h.dimensions == D
    else:
        assert not hasattr(h, "dimensions")
    with pytest.raises(AttributeError):
        h.k = 2
    text = repr(h)
    assert text.startswith(family.name + "(")
    assert "k=4" in text and "repetitions=8" in text
    assert str(h) == text


def test_family_specific_properties():
    assert pp.E2LSH(4, 4, 0.1).quantization_width == np.float32(0.1)
    lattice = pp.LatticeLSH(4, 4, 2.5)
    assert lattice.scaling_factor == 2.5
    np.testing.assert_array_equal(lattice.offset, np.zeros(4))
