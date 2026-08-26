"""BM25 scoring kernels exported through a small C ABI."""

from std.sys.info import simd_width_of

comptime FPtr = Pointer[Float64, AnyOrigin[mut=True]]
comptime IPtr = Pointer[Int64, AnyOrigin[mut=True]]


def fptr(address: Int) -> FPtr:
    return FPtr(unsafe_from_address=address)


def iptr(address: Int) -> IPtr:
    return IPtr(unsafe_from_address=address)


def score_dense_range(
    frequencies: FPtr,
    query_idfs: FPtr,
    document_lengths: FPtr,
    scores: FPtr,
    start: Int,
    end: Int,
    document_count: Int,
    query_count: Int,
    variant: Int,
    k1: Float64,
    b: Float64,
    delta: Float64,
    average_length: Float64,
):
    comptime W = simd_width_of[DType.float64]()
    var document = start
    var vector_end = end - (end - start) % W
    while document < vector_end:
        var length = document_lengths.unsafe_load[width=W](document)
        var norm = 1.0 - b + b * length / average_length
        var score = SIMD[DType.float64, W](0.0)
        for query_index in range(query_count):
            var frequency = frequencies.unsafe_load[width=W](
                query_index * document_count + document
            )
            var idf = query_idfs[unsafe_offset=query_index]
            if variant == 0:
                score += idf * frequency * (k1 + 1.0) / (frequency + k1 * norm)
            elif variant == 1:
                var normalized_frequency = frequency / norm
                score += (
                    idf
                    * frequency
                    * (k1 + 1.0)
                    * (normalized_frequency + delta)
                    / (k1 + normalized_frequency + delta)
                )
            else:
                score += idf * (
                    delta + frequency * (k1 + 1.0) / (k1 * norm + frequency)
                )
        scores.unsafe_store(document, score)
        document += W

    while document < end:
        var norm = (
            1.0
            - b
            + b * document_lengths[unsafe_offset=document] / average_length
        )
        var score = 0.0
        for query_index in range(query_count):
            var frequency = frequencies[
                unsafe_offset=query_index * document_count + document
            ]
            var idf = query_idfs[unsafe_offset=query_index]
            if variant == 0:
                score += idf * frequency * (k1 + 1.0) / (frequency + k1 * norm)
            elif variant == 1:
                var normalized_frequency = frequency / norm
                score += (
                    idf
                    * frequency
                    * (k1 + 1.0)
                    * (normalized_frequency + delta)
                    / (k1 + normalized_frequency + delta)
                )
            else:
                score += idf * (
                    delta + frequency * (k1 + 1.0) / (k1 * norm + frequency)
                )
        scores[unsafe_offset=document] = score
        document += 1


def posting_frequency(
    offsets: IPtr,
    docs: IPtr,
    frequencies: FPtr,
    term: Int,
    document: Int,
) -> Float64:
    var first = Int(offsets[unsafe_offset=term])
    var last = Int(offsets[unsafe_offset=term + 1])
    var posting = first
    var end = last
    while posting < end:
        var middle = posting + (end - posting) // 2
        if Int(docs[unsafe_offset=middle]) < document:
            posting = middle + 1
        else:
            end = middle
    if posting < last and Int(docs[unsafe_offset=posting]) == document:
        return frequencies[unsafe_offset=posting]
    return 0.0


def score_batch_postings_range(
    offsets: IPtr,
    docs: IPtr,
    frequencies: FPtr,
    query_terms: IPtr,
    query_idfs: FPtr,
    norms: FPtr,
    document_ids: IPtr,
    scores: FPtr,
    start: Int,
    end: Int,
    query_count: Int,
    variant: Int,
    k1: Float64,
    delta: Float64,
):
    comptime W = simd_width_of[DType.float64]()
    var batch_index = start
    var vector_end = end - (end - start) % W
    while batch_index < vector_end:
        var documents = document_ids.unsafe_load[width=W](batch_index)
        var norm = SIMD[DType.float64, W](0.0)
        comptime for lane in range(W):
            norm[lane] = norms[unsafe_offset=Int(documents[lane])]
        var score = SIMD[DType.float64, W](0.0)
        for query_index in range(query_count):
            var term = Int(query_terms[unsafe_offset=query_index])
            var idf = query_idfs[unsafe_offset=query_index]
            var frequency = SIMD[DType.float64, W](0.0)
            comptime for lane in range(W):
                frequency[lane] = posting_frequency(
                    offsets,
                    docs,
                    frequencies,
                    term,
                    Int(documents[lane]),
                )
            if variant == 0:
                score += idf * frequency * (k1 + 1.0) / (frequency + k1 * norm)
            elif variant == 1:
                var normalized_frequency = frequency / norm
                score += (
                    idf
                    * frequency
                    * (k1 + 1.0)
                    * (normalized_frequency + delta)
                    / (k1 + normalized_frequency + delta)
                )
            else:
                score += idf * (
                    delta + frequency * (k1 + 1.0) / (k1 * norm + frequency)
                )
        scores.unsafe_store(batch_index, score)
        batch_index += W

    while batch_index < end:
        var document = Int(document_ids[unsafe_offset=batch_index])
        var norm = norms[unsafe_offset=document]
        var score = 0.0
        for query_index in range(query_count):
            var term = Int(query_terms[unsafe_offset=query_index])
            var idf = query_idfs[unsafe_offset=query_index]
            var frequency = posting_frequency(
                offsets, docs, frequencies, term, document
            )
            if variant == 0:
                score += idf * frequency * (k1 + 1.0) / (frequency + k1 * norm)
            elif variant == 1:
                var normalized_frequency = frequency / norm
                score += (
                    idf
                    * frequency
                    * (k1 + 1.0)
                    * (normalized_frequency + delta)
                    / (k1 + normalized_frequency + delta)
                )
            else:
                score += idf * (
                    delta + frequency * (k1 + 1.0) / (k1 * norm + frequency)
                )
        scores[unsafe_offset=batch_index] = score
        batch_index += 1


@export("mrb_score_postings")
def score_postings(
    offsets_address: Int,
    docs_address: Int,
    frequencies_address: Int,
    query_terms_address: Int,
    query_idfs_address: Int,
    norms_address: Int,
    scores_address: Int,
    document_count: Int,
    query_count: Int,
    variant: Int,
    k1: Float64,
    delta: Float64,
) abi("C"):
    var offsets = iptr(offsets_address)
    var docs = iptr(docs_address)
    var frequencies = fptr(frequencies_address)
    var query_terms = iptr(query_terms_address)
    var query_idfs = fptr(query_idfs_address)
    var norms = fptr(norms_address)
    var scores = fptr(scores_address)

    var base = 0.0
    if variant == 2:
        for query_index in range(query_count):
            base += query_idfs[unsafe_offset=query_index] * delta
    for document in range(document_count):
        scores[unsafe_offset=document] = base

    for query_index in range(query_count):
        var term = Int(query_terms[unsafe_offset=query_index])
        var idf = query_idfs[unsafe_offset=query_index]
        for posting in range(
            Int(offsets[unsafe_offset=term]),
            Int(offsets[unsafe_offset=term + 1]),
        ):
            var document = Int(docs[unsafe_offset=posting])
            var frequency = frequencies[unsafe_offset=posting]
            if variant == 0:
                scores[unsafe_offset=document] += (
                    idf
                    * frequency
                    * (k1 + 1.0)
                    / (frequency + k1 * norms[unsafe_offset=document])
                )
            elif variant == 1:
                var normalized_frequency = (
                    frequency / norms[unsafe_offset=document]
                )
                scores[unsafe_offset=document] += (
                    idf
                    * frequency
                    * (k1 + 1.0)
                    * (normalized_frequency + delta)
                    / (k1 + normalized_frequency + delta)
                )
            else:
                scores[unsafe_offset=document] += (
                    idf
                    * frequency
                    * (k1 + 1.0)
                    / (k1 * norms[unsafe_offset=document] + frequency)
                )


@export("mrb_score_batch_postings")
def score_batch_postings(
    offsets_address: Int,
    docs_address: Int,
    frequencies_address: Int,
    query_terms_address: Int,
    query_idfs_address: Int,
    norms_address: Int,
    document_ids_address: Int,
    scores_address: Int,
    batch_count: Int,
    query_count: Int,
    variant: Int,
    k1: Float64,
    delta: Float64,
) abi("C"):
    var offsets = iptr(offsets_address)
    var docs = iptr(docs_address)
    var frequencies = fptr(frequencies_address)
    var query_terms = iptr(query_terms_address)
    var query_idfs = fptr(query_idfs_address)
    var norms = fptr(norms_address)
    var document_ids = iptr(document_ids_address)
    var scores = fptr(scores_address)

    score_batch_postings_range(
        offsets,
        docs,
        frequencies,
        query_terms,
        query_idfs,
        norms,
        document_ids,
        scores,
        0,
        batch_count,
        query_count,
        variant,
        k1,
        delta,
    )


@export("mrb_score_dense")
def score_dense(
    frequencies_address: Int,
    query_idfs_address: Int,
    document_lengths_address: Int,
    scores_address: Int,
    document_count: Int,
    query_count: Int,
    variant: Int,
    k1: Float64,
    b: Float64,
    delta: Float64,
    average_length: Float64,
) abi("C"):
    var frequencies = fptr(frequencies_address)
    var query_idfs = fptr(query_idfs_address)
    var document_lengths = fptr(document_lengths_address)
    var scores = fptr(scores_address)

    score_dense_range(
        frequencies,
        query_idfs,
        document_lengths,
        scores,
        0,
        document_count,
        document_count,
        query_count,
        variant,
        k1,
        b,
        delta,
        average_length,
    )
