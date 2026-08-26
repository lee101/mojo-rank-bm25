# mojo-rank-bm25

`mojo-rank-bm25` is a Mojo implementation of the scoring work in
[`rank-bm25`](https://pypi.org/project/rank-bm25/), exposed through a Python API
with the same class names, constructor signatures, and query methods as
`rank-bm25` 0.2.2.

The corpus index retains the upstream Python attributes while the repeated,
compute-intensive scoring work runs in a compiled Mojo shared library. This is
most useful when one index serves many queries.

## Coverage

| upstream API | status |
| --- | --- |
| `BM25` | Base class and upstream abstract behavior |
| `BM25Okapi` | Supported, including epsilon IDF flooring |
| `BM25L` | Supported |
| `BM25Plus` | Supported |
| `get_scores` | Supported; returns a NumPy array |
| `get_batch_scores` | Supported; returns a list |
| `get_top_n` | Supported, including corpus-length validation |
| `tokenizer=` | Supported |

The project does not tokenize or normalize text unless a `tokenizer` is passed,
matching upstream. `BM25Adpt` and `BM25T` are not covered because they are
commented-out sketches rather than usable classes in `rank-bm25` 0.2.2. Corpus
index construction remains in Python; query scoring is the accelerated subset.
The current build is source-only and Linux-only: it does not publish a wheel,
bundle a prebuilt shared library, provide a command-line interface, or add
features beyond the upstream Python API listed above.

## Install

```bash
pixi install
pixi run build
```

This installs the pinned Mojo nightly, NumPy, pytest, and the real
`rank-bm25==0.2.2` package used by the parity tests. The build produces
`dist/libmojo-rank-bm25.so`.

## Usage

```python
from rank_bm25 import BM25Okapi

documents = [
    "the quick brown fox".split(),
    "the slow brown bear".split(),
    "a fox is quick".split(),
    "a turtle moves slowly".split(),
    "birds fly over water".split(),
]

index = BM25Okapi(documents)
scores = index.get_scores("quick fox".split())
print(scores)
print(index.get_top_n("quick fox".split(), documents, n=2))
```

Run the example from the repository after building:

```bash
pixi run python example.py
```

The normal upstream migration is just the import path already used by
`rank-bm25`; placing this repository's `python/` directory first on
`PYTHONPATH` selects the Mojo-backed implementation. Pixi does this
automatically.

## Correctness

The test suite loads the installed upstream source under a separate module name
and compares both implementations on the same corpora and parameters. It
covers all three algorithms, IDF state, duplicate and unknown query terms,
randomized corpora, batch scores, negative document indices accepted upstream,
top-N results, empty documents, degenerate `NaN` behavior, and public
signatures. Selected-document kernel tests cover exact SIMD widths and scalar
tails for all three scoring variants.

```bash
pixi run build
pixi run test
```

## Performance

Measured with `pixi run bench` on an Intel Xeon E5-2697 v4 at 2.30 GHz, Linux
6.8.0-136-generic, and Python 3.13.14. The full-score cases use 30,000
40-token documents, an 8,000-term vocabulary, and an eight-term query. This is
a sparse retrieval workload: the Mojo path visits postings for matching terms,
whereas upstream creates and scores a dense 30,000-element array for every
query term. The latency cases use 32 12-token documents, a 64-term vocabulary,
and the same query length.

| case | Mojo | rank-bm25 | result |
| --- | ---: | ---: | ---: |
| `BM25Okapi.get_scores` (30k x 40, 8 terms) | 37.0 us | 49.33 ms | 1331.59x faster |
| `BM25L.get_scores` (30k x 40, 8 terms) | 38.0 us | 46.29 ms | 1216.70x faster |
| `BM25Plus.get_scores` (30k x 40, 8 terms) | 36.4 us | 45.42 ms | 1249.24x faster |
| `BM25Okapi.get_batch_scores` (5k docs) | 807.8 us | 11.26 ms | 13.94x faster |
| `BM25Okapi.get_scores` (32 x 12, 8 terms) | 12.5 us | 125.8 us | 10.08x faster |
| `BM25Okapi.get_batch_scores` (6 docs from 32) | 18.1 us | 100.2 us | 5.53x faster |

The large full-score gains come from the indexing strategy as well as compiled
arithmetic; they should not be generalized to every corpus distribution.
For small batches, Mojo binary-searches the sorted postings for each selected
document instead of constructing and copying a dense frequency matrix in
Python. It scores full SIMD-width groups of selected documents and uses a scalar
tail. Larger batches score the sparse corpus once and select the requested
documents. Immutable NumPy buffer addresses are cached with their owners, so
trusted internal FFI calls do not repeat array validation or copy index data.

CPU is the only execution target. The selected-document target is too small for
thread launch overhead, while large batches already use the sparse full-score
path. No GPU path is provided: posting scoring is dominated by sparse and
random-access loads, and the dense fallback also remains below the roughly
2-FLOP-per-byte threshold where device transfer and launch costs could pay off.

## How it works

During construction, Python builds the same `doc_freqs`, `doc_len`, `idf`,
`avgdl`, and related public attributes as upstream. It also creates a
term-major compressed posting layout:

```text
offsets[int64] -> doc_ids[int64] + term_frequencies[float64]
document_norms[float64]
```

`get_scores` maps query tokens to term IDs and IDF values, then makes one
ctypes call. Mojo initializes the output and walks only the relevant posting
ranges, applying the exact Okapi, L, or Plus formula. `get_batch_scores` chooses
between SIMD selected-posting scoring for small selections and sparse scoring
followed by indexed selection for larger batches.

All memory is owned by NumPy. Across the C ABI, buffers are passed as integer
addresses and rebuilt as `UnsafePointer[..., AnyOrigin[mut=True]]` inside the
non-parametric `@export` functions. The Mojo library never retains a pointer or
allocates caller-visible memory. Public low-level wrappers validate arbitrary
buffers; model-owned immutable buffers use cached addresses and remain strongly
referenced for the lifetime of the index.

## License

MIT
