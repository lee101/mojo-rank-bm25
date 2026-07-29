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
signatures.

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
query term.

| case | Mojo | rank-bm25 | result |
| --- | ---: | ---: | ---: |
| `BM25Okapi.get_scores` (30k x 40, 8 terms) | 185.2 us | 99.62 ms | 537.83x faster |
| `BM25L.get_scores` (30k x 40, 8 terms) | 96.4 us | 96.05 ms | 996.60x faster |
| `BM25Plus.get_scores` (30k x 40, 8 terms) | 147.0 us | 97.09 ms | 660.28x faster |
| `BM25Okapi.get_batch_scores` (5k docs) | 1.37 ms | 11.66 ms | 8.49x faster |

The large full-score gains come from the indexing strategy as well as compiled
arithmetic; they should not be generalized to every corpus distribution.
For small batches, scoring keeps the dense path to avoid full-corpus work. Once
Python dictionary gathering becomes more expensive, it scores the sparse
postings and selects the requested documents instead. Document lengths are
cached as a NumPy buffer rather than copied from a Python list on every call.

The dense fallback processes full SIMD-width document blocks with a scalar tail
and splits sufficiently large inputs into independent tasks. CPU is the only
execution target; this project does not provide a GPU kernel.

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
between SIMD dense scoring for small selections and sparse scoring followed by
indexed selection for larger batches.

All memory is owned by NumPy. Across the C ABI, buffers are passed as integer
addresses and rebuilt as `UnsafePointer[..., AnyOrigin[mut=True]]` inside the
non-parametric `@export` functions. The Mojo library never retains a pointer or
allocates caller-visible memory.

## License

MIT
