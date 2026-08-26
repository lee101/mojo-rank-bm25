"""A Mojo-accelerated, API-compatible subset of rank-bm25."""

from __future__ import annotations

import math
from multiprocessing import Pool, cpu_count

import numpy as np

from ._lib import (
    _score_batch_postings_unchecked,
    _score_postings_unchecked,
    score_dense,
)

__all__ = ["BM25", "BM25Okapi", "BM25L", "BM25Plus"]


class BM25:
    def __init__(self, corpus, tokenizer=None):
        self.corpus_size = 0
        self.avgdl = 0
        self.doc_freqs = []
        self.idf = {}
        self.doc_len = []
        self.tokenizer = tokenizer

        if tokenizer:
            corpus = self._tokenize_corpus(corpus)

        nd = self._initialize(corpus)
        self._calc_idf(nd)
        self._build_postings()

    def _initialize(self, corpus):
        nd = {}
        num_doc = 0
        for document in corpus:
            self.doc_len.append(len(document))
            num_doc += len(document)

            frequencies = {}
            for word in document:
                if word not in frequencies:
                    frequencies[word] = 0
                frequencies[word] += 1
            self.doc_freqs.append(frequencies)

            for word in frequencies:
                try:
                    nd[word] += 1
                except KeyError:
                    nd[word] = 1

            self.corpus_size += 1

        self.avgdl = num_doc / self.corpus_size
        return nd

    def _tokenize_corpus(self, corpus):
        with Pool(cpu_count()) as pool:
            return pool.map(self.tokenizer, corpus)

    def _calc_idf(self, nd):
        raise NotImplementedError()

    def get_scores(self, query):
        raise NotImplementedError()

    def get_batch_scores(self, query, doc_ids):
        raise NotImplementedError()

    def get_top_n(self, query, documents, n=5):
        assert self.corpus_size == len(
            documents
        ), "The documents given don't match the index corpus!"
        scores = self.get_scores(query)
        top_n = np.argsort(scores)[::-1][:n]
        return [documents[i] for i in top_n]

    def _build_postings(self):
        self._term_ids = {term: index for index, term in enumerate(self.idf)}
        docs_by_term = [[] for _ in self.idf]
        frequencies_by_term = [[] for _ in self.idf]
        for document, frequencies in enumerate(self.doc_freqs):
            for term, frequency in frequencies.items():
                term_id = self._term_ids[term]
                docs_by_term[term_id].append(document)
                frequencies_by_term[term_id].append(float(frequency))
        offsets = [0]
        posting_docs = []
        posting_frequencies = []
        for term_docs, term_frequencies in zip(docs_by_term, frequencies_by_term):
            posting_docs.extend(term_docs)
            posting_frequencies.extend(term_frequencies)
            offsets.append(len(posting_docs))
        self._offsets = np.ascontiguousarray(offsets, dtype=np.int64)
        self._posting_docs = np.ascontiguousarray(posting_docs, dtype=np.int64)
        self._posting_frequencies = np.ascontiguousarray(
            posting_frequencies, dtype=np.float64
        )
        lengths = np.ascontiguousarray(self.doc_len, dtype=np.float64)
        self._document_lengths = lengths
        if self.avgdl == 0:
            self._norms = np.zeros(self.corpus_size, dtype=np.float64)
        else:
            self._norms = np.ascontiguousarray(
                1.0 - self.b + self.b * lengths / self.avgdl, dtype=np.float64
            )
        self._has_zero_norm = bool(np.any(self._norms == 0))
        for buffer in (
            self._offsets,
            self._posting_docs,
            self._posting_frequencies,
            self._document_lengths,
            self._norms,
        ):
            buffer.flags.writeable = False
        self._posting_addresses = (
            int(self._offsets.ctypes.data),
            int(self._posting_docs.ctypes.data),
            int(self._posting_frequencies.ctypes.data),
            int(self._norms.ctypes.data),
        )

    def _score(self, query, variant):
        query = list(query)
        if self.avgdl == 0 or self._has_zero_norm:
            return self._score_dense(query, list(range(self.corpus_size)), variant)

        known = [term for term in query if term in self._term_ids]
        if not known:
            return np.zeros(self.corpus_size, dtype=np.float64)
        term_ids = np.ascontiguousarray(
            [self._term_ids[term] for term in known], dtype=np.int64
        )
        idfs = np.ascontiguousarray([self.idf[term] for term in known], dtype=np.float64)
        scores = np.empty(self.corpus_size, dtype=np.float64)
        _score_postings_unchecked(
            self._posting_addresses[0],
            self._posting_addresses[1],
            self._posting_addresses[2],
            term_ids,
            idfs,
            self._posting_addresses[3],
            scores,
            variant,
            self.k1,
            self.delta,
        )
        return scores

    def _score_dense(self, query, doc_ids, variant):
        query = list(query)
        assert all(di < len(self.doc_freqs) for di in doc_ids)
        document_lengths = np.ascontiguousarray(
            self._document_lengths[doc_ids], dtype=np.float64
        )
        count = len(doc_ids)
        scores = np.zeros(count, dtype=np.float64)
        if not query or not count:
            return scores
        frequencies = np.ascontiguousarray(
            [
                self.doc_freqs[document].get(term) or 0
                for term in query
                for document in doc_ids
            ],
            dtype=np.float64,
        )
        idfs = np.ascontiguousarray(
            [self.idf.get(term) or 0 for term in query], dtype=np.float64
        )
        score_dense(
            frequencies,
            idfs,
            document_lengths,
            scores,
            variant,
            self.k1,
            self.b,
            self.delta,
            self.avgdl,
        )
        return scores

    def _score_batch(self, query, doc_ids, variant):
        query = list(query)
        assert all(di < len(self.doc_freqs) for di in doc_ids)
        count = len(doc_ids)
        if not count:
            return []
        if not query:
            return [0.0] * count
        dense_lookups = count * len(query)
        sparse_threshold = max(512, self.corpus_size // 32)
        if dense_lookups < sparse_threshold:
            known = [term for term in query if term in self._term_ids]
            if not known:
                return [0.0] * count
            has_negative = False
            for document in doc_ids:
                if document < -self.corpus_size:
                    raise IndexError("document index is out of range")
                has_negative = has_negative or document < 0
            indices = np.ascontiguousarray(doc_ids, dtype=np.int64)
            if has_negative:
                indices = indices.copy()
                indices[indices < 0] += self.corpus_size
            term_ids = np.ascontiguousarray(
                [self._term_ids[term] for term in known], dtype=np.int64
            )
            idfs = np.ascontiguousarray(
                [self.idf[term] for term in known], dtype=np.float64
            )
            scores = np.empty(count, dtype=np.float64)
            _score_batch_postings_unchecked(
                self._posting_addresses[0],
                self._posting_addresses[1],
                self._posting_addresses[2],
                term_ids,
                idfs,
                self._posting_addresses[3],
                indices,
                scores,
                variant,
                self.k1,
                self.delta,
            )
            return scores.tolist()
        indices = np.asarray(doc_ids)
        return self._score(query, variant)[indices].tolist()


class BM25Okapi(BM25):
    def __init__(self, corpus, tokenizer=None, k1=1.5, b=0.75, epsilon=0.25):
        self.k1 = k1
        self.b = b
        self.epsilon = epsilon
        self.delta = 0.0
        super().__init__(corpus, tokenizer)

    def _calc_idf(self, nd):
        idf_sum = 0
        negative_idfs = []
        for word, frequency in nd.items():
            idf = math.log(self.corpus_size - frequency + 0.5) - math.log(
                frequency + 0.5
            )
            self.idf[word] = idf
            idf_sum += idf
            if idf < 0:
                negative_idfs.append(word)
        self.average_idf = idf_sum / len(self.idf)

        eps = self.epsilon * self.average_idf
        for word in negative_idfs:
            self.idf[word] = eps

    def get_scores(self, query):
        return self._score(query, 0)

    def get_batch_scores(self, query, doc_ids):
        return self._score_batch(query, doc_ids, 0)


class BM25L(BM25):
    def __init__(self, corpus, tokenizer=None, k1=1.5, b=0.75, delta=0.5):
        self.k1 = k1
        self.b = b
        self.delta = delta
        super().__init__(corpus, tokenizer)

    def _calc_idf(self, nd):
        for word, frequency in nd.items():
            self.idf[word] = math.log(self.corpus_size + 1) - math.log(frequency + 0.5)

    def get_scores(self, query):
        return self._score(query, 1)

    def get_batch_scores(self, query, doc_ids):
        return self._score_batch(query, doc_ids, 1)


class BM25Plus(BM25):
    def __init__(self, corpus, tokenizer=None, k1=1.5, b=0.75, delta=1):
        self.k1 = k1
        self.b = b
        self.delta = delta
        super().__init__(corpus, tokenizer)

    def _calc_idf(self, nd):
        for word, frequency in nd.items():
            self.idf[word] = math.log((self.corpus_size + 1) / frequency)

    def get_scores(self, query):
        return self._score(query, 2)

    def get_batch_scores(self, query, doc_ids):
        return self._score_batch(query, doc_ids, 2)
