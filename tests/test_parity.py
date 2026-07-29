"""Behavioral and numerical parity with rank-bm25 0.2.2."""

import inspect

import numpy as np
import pytest

import rank_bm25 as mojo_bm25
from rank_bm25._lib import score_dense


CORPUS = [
    "It is quite windy in London".lower().split(),
    "London is the capital of the United Kingdom".lower().split(),
    "The capital of France is Paris".lower().split(),
    "Mojo makes systems programming productive".lower().split(),
    [],
    ["london", "london", "ranking", "bm25"],
]

CLASSES = ["BM25Okapi", "BM25L", "BM25Plus"]


@pytest.mark.parametrize("name", CLASSES)
def test_default_scores(upstream, name):
    ours = getattr(mojo_bm25, name)(CORPUS)
    theirs = getattr(upstream, name)(CORPUS)
    for query in (
        ["capital", "london"],
        ["london", "london", "missing"],
        ["mojo", "programming"],
        ["missing"],
        [],
    ):
        assert np.allclose(
            ours.get_scores(query), theirs.get_scores(query), rtol=1e-14, atol=1e-14
        )


@pytest.mark.parametrize(
    ("name", "kwargs"),
    [
        ("BM25Okapi", {"k1": 0.9, "b": 0.2, "epsilon": 0.6}),
        ("BM25L", {"k1": 2.1, "b": 0.4, "delta": 0.8}),
        ("BM25Plus", {"k1": 1.1, "b": 1.0, "delta": 2.5}),
    ],
)
def test_custom_parameters(upstream, name, kwargs):
    ours = getattr(mojo_bm25, name)(CORPUS, **kwargs)
    theirs = getattr(upstream, name)(CORPUS, **kwargs)
    with np.errstate(all="ignore"):
        actual = ours.get_scores(["the", "capital", "london"])
        expected = theirs.get_scores(["the", "capital", "london"])
    assert np.allclose(actual, expected, rtol=1e-14, atol=1e-14, equal_nan=True)


@pytest.mark.parametrize("name", CLASSES)
def test_batch_scores(upstream, name):
    ours = getattr(mojo_bm25, name)(CORPUS)
    theirs = getattr(upstream, name)(CORPUS)
    doc_ids = [5, 0, 5, -2, 2]
    actual = ours.get_batch_scores(["london", "capital", "london"], doc_ids)
    expected = theirs.get_batch_scores(["london", "capital", "london"], doc_ids)
    assert isinstance(actual, list)
    assert actual == pytest.approx(expected, rel=1e-14, abs=1e-14)
    assert ours.get_batch_scores([], []) == theirs.get_batch_scores([], [])
    assert ours.get_batch_scores([], [0, 2]) == theirs.get_batch_scores([], [0, 2])


@pytest.mark.parametrize("name", CLASSES)
def test_random_corpus_parity(upstream, name):
    rng = np.random.default_rng(42)
    vocabulary = [f"t{i}" for i in range(37)]
    corpus = [
        [vocabulary[i] for i in rng.integers(0, len(vocabulary), size=int(length))]
        for length in rng.integers(1, 45, size=250)
    ]
    query = [vocabulary[i] for i in rng.integers(0, len(vocabulary), size=15)]
    ours = getattr(mojo_bm25, name)(corpus)
    theirs = getattr(upstream, name)(corpus)
    assert np.allclose(
        ours.get_scores(query), theirs.get_scores(query), rtol=2e-14, atol=2e-14
    )


def test_okapi_idf_and_attributes(upstream):
    ours = mojo_bm25.BM25Okapi(CORPUS)
    theirs = upstream.BM25Okapi(CORPUS)
    assert ours.corpus_size == theirs.corpus_size
    assert ours.avgdl == theirs.avgdl
    assert ours.doc_len == theirs.doc_len
    assert ours.doc_freqs == theirs.doc_freqs
    assert ours.idf == theirs.idf
    assert ours.average_idf == theirs.average_idf
    common_corpus = [["common", str(index)] for index in range(5)] + [["rare"]]
    floored = mojo_bm25.BM25Okapi(common_corpus)
    assert floored.idf["common"] == pytest.approx(
        floored.epsilon * floored.average_idf, abs=0
    )


@pytest.mark.parametrize("name", CLASSES)
def test_top_n(upstream, name):
    documents = [" ".join(document) for document in CORPUS]
    ours = getattr(mojo_bm25, name)(CORPUS)
    theirs = getattr(upstream, name)(CORPUS)
    assert ours.get_top_n(["capital", "london"], documents, n=4) == theirs.get_top_n(
        ["capital", "london"], documents, n=4
    )
    with pytest.raises(AssertionError, match="don't match"):
        ours.get_top_n(["capital"], documents[:-1])


@pytest.mark.parametrize("name", ["BM25L", "BM25Plus"])
def test_all_empty_documents(upstream, name):
    corpus = [[], [], []]
    ours = getattr(mojo_bm25, name)(corpus)
    theirs = getattr(upstream, name)(corpus)
    with np.errstate(all="ignore"):
        actual = ours.get_scores(["unknown"])
        expected = theirs.get_scores(["unknown"])
    assert np.array_equal(np.isnan(actual), np.isnan(expected))


def test_base_class_is_abstract_by_behavior():
    with pytest.raises(NotImplementedError):
        mojo_bm25.BM25([["term"]])


def test_public_signatures(upstream):
    for name in ["BM25", *CLASSES]:
        ours = getattr(mojo_bm25, name)
        theirs = getattr(upstream, name)
        assert inspect.signature(ours.__init__) == inspect.signature(theirs.__init__)
        assert inspect.signature(ours.get_scores) == inspect.signature(theirs.get_scores)
        assert inspect.signature(ours.get_batch_scores) == inspect.signature(
            theirs.get_batch_scores
        )
        assert inspect.signature(ours.get_top_n) == inspect.signature(theirs.get_top_n)


def test_invalid_batch_document_id_matches_upstream():
    model = mojo_bm25.BM25Okapi(CORPUS)
    with pytest.raises(AssertionError):
        model.get_batch_scores(["london"], [len(CORPUS)])


def _dense_kernel_scores(document_count):
    rng = np.random.default_rng(7)
    query_count = 3
    frequencies = np.ascontiguousarray(
        rng.integers(0, 6, size=(query_count, document_count)), dtype=np.float64
    )
    idfs = np.ascontiguousarray([0.3, 1.1, 2.4], dtype=np.float64)
    lengths = np.ascontiguousarray(
        rng.integers(1, 80, size=document_count), dtype=np.float64
    )
    scores = np.empty(document_count, dtype=np.float64)
    k1 = 1.5
    b = 0.75
    average_length = 40.0
    score_dense(
        frequencies,
        idfs,
        lengths,
        scores,
        0,
        k1,
        b,
        0.0,
        average_length,
    )
    norms = 1.0 - b + b * lengths / average_length
    expected = np.zeros(document_count, dtype=np.float64)
    for query_index in range(query_count):
        frequency = frequencies[query_index]
        expected += (
            idfs[query_index]
            * frequency
            * (k1 + 1.0)
            / (frequency + k1 * norms)
        )
    return scores, expected


@pytest.mark.parametrize("document_count", [7, 16_383, 16_384, 16_389])
def test_dense_kernel_simd_tail_and_parallel_threshold(document_count):
    actual, expected = _dense_kernel_scores(document_count)
    assert np.allclose(actual, expected, rtol=1e-14, atol=1e-14)


def test_ffi_rejects_wrong_dtype_shape_layout_and_read_only_output():
    frequencies = np.ones((2, 3), dtype=np.float64)
    idfs = np.ones(2, dtype=np.float64)
    lengths = np.ones(3, dtype=np.float64)
    scores = np.empty(3, dtype=np.float64)

    with pytest.raises(TypeError, match="dtype float64"):
        score_dense(
            frequencies.astype(np.float32), idfs, lengths, scores, 0, 1.5, 0.75, 0, 1
        )
    with pytest.raises(ValueError, match="exactly 6"):
        score_dense(
            frequencies[:, :2].copy(), idfs, lengths, scores, 0, 1.5, 0.75, 0, 1
        )
    with pytest.raises(ValueError, match="C-contiguous"):
        score_dense(
            frequencies.T, idfs, lengths, scores, 0, 1.5, 0.75, 0, 1
        )
    scores.flags.writeable = False
    with pytest.raises(ValueError, match="writable"):
        score_dense(frequencies, idfs, lengths, scores, 0, 1.5, 0.75, 0, 1)


def whitespace_tokenizer(text):
    return text.lower().split()


def test_tokenizer_matches_upstream(upstream):
    corpus = ["One TWO", "two three", "THREE three"]
    ours = mojo_bm25.BM25Okapi(corpus, tokenizer=whitespace_tokenizer)
    theirs = upstream.BM25Okapi(corpus, tokenizer=whitespace_tokenizer)
    assert np.allclose(ours.get_scores(["two"]), theirs.get_scores(["two"]))


@pytest.mark.parametrize("name", CLASSES)
def test_batch_strategy_thresholds(upstream, monkeypatch, name):
    corpus = [[f"term_{index % 13}", "shared"] for index in range(250)]
    query = ["term_1", "shared", "term_7"]
    ours = getattr(mojo_bm25, name)(corpus)
    theirs = getattr(upstream, name)(corpus)

    def unexpected_sparse(*args, **kwargs):
        raise AssertionError("small batches should stay dense")

    monkeypatch.setattr(ours, "_score", unexpected_sparse)
    assert ours.get_batch_scores(query, [3]) == pytest.approx(
        theirs.get_batch_scores(query, [3]), rel=1e-14, abs=1e-14
    )

    ours = getattr(mojo_bm25, name)(corpus)

    def unexpected_dense(*args, **kwargs):
        raise AssertionError("large batches should use sparse scoring")

    monkeypatch.setattr(ours, "_score_dense", unexpected_dense)
    doc_ids = list(range(200))
    assert ours.get_batch_scores(query, doc_ids) == pytest.approx(
        theirs.get_batch_scores(query, doc_ids), rel=1e-14, abs=1e-14
    )
