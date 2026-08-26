"""ctypes bindings for the Mojo BM25 scoring kernels."""

from __future__ import annotations

import ctypes
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LIB = os.environ.get("MOJO_RANK_BM25_LIB") or os.path.join(
    ROOT, "dist", "libmojo-rank-bm25.so"
)

I = ctypes.c_int64
F = ctypes.c_double

_SIGNATURES = {
    "mrb_score_postings": ([I] * 10 + [F, F], None),
    "mrb_score_batch_postings": ([I] * 11 + [F, F], None),
    "mrb_score_dense": ([I] * 7 + [F, F, F, F], None),
}

_library: ctypes.CDLL | None = None


def _load_library() -> ctypes.CDLL:
    global _library
    if _library is None:
        if not os.path.exists(LIB):
            raise RuntimeError("Mojo library not built; run `pixi run build`")
        _library = ctypes.CDLL(LIB)
        for name, (argtypes, restype) in _SIGNATURES.items():
            function = getattr(_library, name)
            function.argtypes = argtypes
            function.restype = restype
    return _library


def _buffer(
    array: np.ndarray,
    *,
    name: str,
    dtype: np.dtype,
    size: int,
    writable: bool = False,
) -> int:
    if not isinstance(array, np.ndarray):
        raise TypeError(f"{name} must be a NumPy array")
    if array.dtype != dtype:
        raise TypeError(f"{name} must have dtype {dtype}")
    if array.size != size:
        raise ValueError(f"{name} must contain exactly {size} elements")
    if not array.flags.c_contiguous or not array.flags.aligned:
        raise ValueError(f"{name} must be an aligned C-contiguous array")
    if writable and not array.flags.writeable:
        raise ValueError(f"{name} must be writable")
    if size == 0 or array.ctypes.data == 0:
        raise ValueError(f"{name} must be non-empty and non-null")
    return int(array.ctypes.data)


def score_postings(
    offsets: np.ndarray,
    docs: np.ndarray,
    frequencies: np.ndarray,
    query_terms: np.ndarray,
    query_idfs: np.ndarray,
    norms: np.ndarray,
    scores: np.ndarray,
    variant: int,
    k1: float,
    delta: float,
) -> None:
    """Validate and synchronously call the sparse C ABI kernel."""
    document_count = scores.size
    query_count = query_terms.size
    posting_count = frequencies.size
    if query_count == 0 or document_count == 0 or posting_count == 0:
        raise ValueError("sparse kernel buffers must be non-empty")
    pointers = (
        _buffer(
            offsets,
            name="offsets",
            dtype=np.dtype(np.int64),
            size=offsets.size,
        ),
        _buffer(docs, name="docs", dtype=np.dtype(np.int64), size=posting_count),
        _buffer(
            frequencies,
            name="frequencies",
            dtype=np.dtype(np.float64),
            size=posting_count,
        ),
        _buffer(
            query_terms,
            name="query_terms",
            dtype=np.dtype(np.int64),
            size=query_count,
        ),
        _buffer(
            query_idfs,
            name="query_idfs",
            dtype=np.dtype(np.float64),
            size=query_count,
        ),
        _buffer(
            norms,
            name="norms",
            dtype=np.dtype(np.float64),
            size=document_count,
        ),
        _buffer(
            scores,
            name="scores",
            dtype=np.dtype(np.float64),
            size=document_count,
            writable=True,
        ),
    )
    if offsets.size < 2:
        raise ValueError("offsets must contain at least two elements")
    if np.any(query_terms < 0) or np.any(query_terms >= offsets.size - 1):
        raise ValueError("query_terms contains an out-of-range term id")
    starts = offsets[query_terms]
    ends = offsets[query_terms + 1]
    if np.any(starts < 0) or np.any(starts > ends) or np.any(ends > posting_count):
        raise ValueError("offsets contains an invalid posting range")
    # The tuple and array arguments remain strongly referenced until CDLL returns.
    _load_library().mrb_score_postings(
        *pointers,
        document_count,
        query_count,
        variant,
        k1,
        delta,
    )


def _score_postings_unchecked(
    offsets_address: int,
    docs_address: int,
    frequencies_address: int,
    query_terms: np.ndarray,
    query_idfs: np.ndarray,
    norms_address: int,
    scores: np.ndarray,
    variant: int,
    k1: float,
    delta: float,
) -> None:
    _load_library().mrb_score_postings(
        offsets_address,
        docs_address,
        frequencies_address,
        int(query_terms.ctypes.data),
        int(query_idfs.ctypes.data),
        norms_address,
        int(scores.ctypes.data),
        scores.size,
        query_terms.size,
        variant,
        k1,
        delta,
    )


def score_dense(
    frequencies: np.ndarray,
    query_idfs: np.ndarray,
    document_lengths: np.ndarray,
    scores: np.ndarray,
    variant: int,
    k1: float,
    b: float,
    delta: float,
    average_length: float,
) -> None:
    """Validate and synchronously call the dense C ABI kernel."""
    document_count = scores.size
    query_count = query_idfs.size
    if document_count == 0 or query_count == 0:
        raise ValueError("dense kernel buffers must be non-empty")
    pointers = (
        _buffer(
            frequencies,
            name="frequencies",
            dtype=np.dtype(np.float64),
            size=document_count * query_count,
        ),
        _buffer(
            query_idfs,
            name="query_idfs",
            dtype=np.dtype(np.float64),
            size=query_count,
        ),
        _buffer(
            document_lengths,
            name="document_lengths",
            dtype=np.dtype(np.float64),
            size=document_count,
        ),
        _buffer(
            scores,
            name="scores",
            dtype=np.dtype(np.float64),
            size=document_count,
            writable=True,
        ),
    )
    _load_library().mrb_score_dense(
        *pointers,
        document_count,
        query_count,
        variant,
        k1,
        b,
        delta,
        average_length,
    )


def score_batch_postings(
    offsets: np.ndarray,
    docs: np.ndarray,
    frequencies: np.ndarray,
    query_terms: np.ndarray,
    query_idfs: np.ndarray,
    norms: np.ndarray,
    document_ids: np.ndarray,
    scores: np.ndarray,
    variant: int,
    k1: float,
    delta: float,
) -> None:
    """Score selected documents directly from sorted posting lists."""
    batch_count = document_ids.size
    query_count = query_terms.size
    posting_count = frequencies.size
    if batch_count == 0 or query_count == 0 or posting_count == 0:
        raise ValueError("batch posting kernel buffers must be non-empty")
    pointers = (
        _buffer(offsets, name="offsets", dtype=np.dtype(np.int64), size=offsets.size),
        _buffer(docs, name="docs", dtype=np.dtype(np.int64), size=posting_count),
        _buffer(
            frequencies,
            name="frequencies",
            dtype=np.dtype(np.float64),
            size=posting_count,
        ),
        _buffer(
            query_terms,
            name="query_terms",
            dtype=np.dtype(np.int64),
            size=query_count,
        ),
        _buffer(
            query_idfs,
            name="query_idfs",
            dtype=np.dtype(np.float64),
            size=query_count,
        ),
        _buffer(norms, name="norms", dtype=np.dtype(np.float64), size=norms.size),
        _buffer(
            document_ids,
            name="document_ids",
            dtype=np.dtype(np.int64),
            size=batch_count,
        ),
        _buffer(
            scores,
            name="scores",
            dtype=np.dtype(np.float64),
            size=batch_count,
            writable=True,
        ),
    )
    if offsets.size < 2:
        raise ValueError("offsets must contain at least two elements")
    if np.any(query_terms < 0) or np.any(query_terms >= offsets.size - 1):
        raise ValueError("query_terms contains an out-of-range term id")
    if np.any(document_ids < 0) or np.any(document_ids >= norms.size):
        raise ValueError("document_ids contains an out-of-range document id")
    _load_library().mrb_score_batch_postings(
        *pointers,
        batch_count,
        query_count,
        variant,
        k1,
        delta,
    )


def _score_batch_postings_unchecked(
    offsets_address: int,
    docs_address: int,
    frequencies_address: int,
    query_terms: np.ndarray,
    query_idfs: np.ndarray,
    norms_address: int,
    document_ids: np.ndarray,
    scores: np.ndarray,
    variant: int,
    k1: float,
    delta: float,
) -> None:
    _load_library().mrb_score_batch_postings(
        offsets_address,
        docs_address,
        frequencies_address,
        int(query_terms.ctypes.data),
        int(query_idfs.ctypes.data),
        norms_address,
        int(document_ids.ctypes.data),
        int(scores.ctypes.data),
        document_ids.size,
        query_terms.size,
        variant,
        k1,
        delta,
    )
