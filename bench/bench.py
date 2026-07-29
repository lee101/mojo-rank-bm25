"""Benchmark Mojo scoring against rank-bm25 on identical corpora."""

from __future__ import annotations

import importlib.metadata
import importlib.util
import math
import os
import platform
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "python"))

import rank_bm25 as mojo_bm25  # noqa: E402


def load_upstream():
    path = importlib.metadata.distribution("rank-bm25").locate_file("rank_bm25.py")
    spec = importlib.util.spec_from_file_location("_benchmark_upstream_rank_bm25", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


upstream = load_upstream()


def machine() -> str:
    cpu = platform.processor()
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as handle:
            cpu = next(
                line.split(":", 1)[1].strip()
                for line in handle
                if line.startswith("model name")
            )
    except (OSError, StopIteration):
        pass
    return f"{cpu}; {platform.system()} {platform.release()}; Python {platform.python_version()}"


def time_call(function, minimum=0.2, repeats=5):
    loops = 1
    while True:
        start = time.perf_counter()
        for _ in range(loops):
            function()
        elapsed = time.perf_counter() - start
        if elapsed >= minimum:
            break
        loops *= 2
    best = math.inf
    for _ in range(repeats):
        start = time.perf_counter()
        for _ in range(loops):
            function()
        best = min(best, (time.perf_counter() - start) / loops)
    return best


def make_corpus(document_count=30_000, document_length=40, vocabulary_size=8_000):
    rng = np.random.default_rng(123)
    vocabulary = [f"term_{index}" for index in range(vocabulary_size)]
    values = rng.integers(
        0, vocabulary_size, size=(document_count, document_length), dtype=np.int32
    )
    corpus = [[vocabulary[index] for index in row] for row in values]
    query = [vocabulary[index] for index in (3, 71, 907, 1337, 4099, 7171, 3, 907)]
    return corpus, query


def format_time(seconds):
    if seconds < 0.001:
        return f"{seconds * 1e6:.1f} us"
    return f"{seconds * 1e3:.2f} ms"


def benchmark_scores(name, corpus, query):
    ours = getattr(mojo_bm25, name)(corpus)
    theirs = getattr(upstream, name)(corpus)
    ours.get_scores(query)
    theirs.get_scores(query)
    return time_call(lambda: ours.get_scores(query)), time_call(
        lambda: theirs.get_scores(query)
    )


def benchmark_batch(name, corpus, query):
    ours = getattr(mojo_bm25, name)(corpus)
    theirs = getattr(upstream, name)(corpus)
    doc_ids = list(range(0, len(corpus), 6))
    ours.get_batch_scores(query, doc_ids)
    theirs.get_batch_scores(query, doc_ids)
    return time_call(lambda: ours.get_batch_scores(query, doc_ids)), time_call(
        lambda: theirs.get_batch_scores(query, doc_ids)
    )


def main():
    corpus, query = make_corpus()
    cases = [
        (
            "BM25Okapi.get_scores (30k x 40, 8 terms)",
            lambda: benchmark_scores("BM25Okapi", corpus, query),
        ),
        (
            "BM25L.get_scores (30k x 40, 8 terms)",
            lambda: benchmark_scores("BM25L", corpus, query),
        ),
        (
            "BM25Plus.get_scores (30k x 40, 8 terms)",
            lambda: benchmark_scores("BM25Plus", corpus, query),
        ),
        (
            "BM25Okapi.get_batch_scores (5k docs)",
            lambda: benchmark_batch("BM25Okapi", corpus, query),
        ),
    ]

    print(f"Machine: {machine()}")
    print()
    print("| case | Mojo | rank-bm25 | result |")
    print("| --- | ---: | ---: | ---: |")
    for label, run in cases:
        mojo_time, upstream_time = run()
        ratio = upstream_time / mojo_time
        result = (
            f"{ratio:.2f}x faster"
            if ratio >= 1
            else f"{1.0 / ratio:.2f}x slower"
        )
        print(
            f"| {label} | {format_time(mojo_time)} | "
            f"{format_time(upstream_time)} | {result} |"
        )


if __name__ == "__main__":
    main()
