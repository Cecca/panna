from ._panna_impl import *
from .datasets import *
from .knn import *
from .mst import *

from ._panna_impl import MinHash as _MinHash, MinHash1Bit as _MinHash1Bit


def _set_tokens(s):
    """Converts one set to a one-dimensional int64 array of tokens."""
    import numpy as np

    if isinstance(s, (np.ndarray, list, tuple, range)):
        tokens = np.asarray(s)
    else:
        tokens = np.fromiter(s, dtype=np.int64)
    if tokens.ndim != 1:
        raise ValueError("each set must be a one-dimensional sequence of integers")
    if tokens.size == 0:
        return np.empty(0, dtype=np.int64)
    if tokens.dtype.kind not in "iu":
        raise TypeError(f"set tokens must be integers, got dtype {tokens.dtype}")
    return tokens.astype(np.int64, copy=False)


def _sets_to_csr(sets):
    """Returns the `(indptr, indices)` arrays of a collection of sets."""
    import numpy as np

    # scipy.sparse matrices and arrays: rows are sets, columns are tokens.
    if hasattr(sets, "tocsr"):
        if getattr(sets, "format", None) != "csr":
            sets = sets.tocsr()
        return sets.indptr, sets.indices

    chunks = [_set_tokens(s) for s in sets]
    indptr = np.zeros(len(chunks) + 1, dtype=np.int64)
    np.cumsum([len(c) for c in chunks], out=indptr[1:])
    indices = np.concatenate(chunks) if chunks else np.empty(0, dtype=np.int64)
    return indptr, indices


def _hash_sets(self, sets, packed=False):
    """Hashes a collection of sets of integer tokens.

    A convenience over `hash(indptr, indices, packed)`, which this method
    calls after building the CSR arrays.

    Args:
      sets: either a scipy.sparse matrix or array, where row `i` is set `i`
        and its tokens are the column indices of the stored entries (formats
        other than CSR are converted with `.tocsr()`; every stored entry
        counts, explicit zeros included, so call `eliminate_zeros()` first if
        needed), or an iterable of sets, each an iterable of integer tokens
        in [0, 2**32) (lists, tuples, sets, NumPy arrays, ...).
      packed (bool, default False): return one key per repetition rather
        than the individual codes.

    Returns:
      The same as `hash`: the codes of shape (n, repetitions, k), or the
      packed keys of shape (n, repetitions). The empty set hashes to token 0.

    Raises:
      ValueError: if a set is not one-dimensional or a token is out of range.
      TypeError: if a set holds non-integer values.
    """
    indptr, indices = _sets_to_csr(sets)
    return self.hash(indptr, indices, packed=packed)


for _cls in (_MinHash, _MinHash1Bit):
    _cls.hash_sets = _hash_sets
del _cls
