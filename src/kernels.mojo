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
                score += (
                    idf * frequency * (k1 + 1.0)
                    / (frequency + k1 * norm)
                )
            elif variant == 1:
                var normalized_frequency = frequency / norm
                score += (
                    idf * frequency * (k1 + 1.0)
                    * (normalized_frequency + delta)
                    / (k1 + normalized_frequency + delta)
                )
            else:
                score += (
                    idf * (
                        delta
                        + frequency * (k1 + 1.0)
                        / (k1 * norm + frequency)
                    )
                )
        scores.unsafe_store(document, score)
        document += W

    while document < end:
        var norm = (
            1.0 - b
            + b * document_lengths[unsafe_offset=document] / average_length
        )
        var score = 0.0
        for query_index in range(query_count):
            var frequency = frequencies[
                unsafe_offset=query_index * document_count + document
            ]
            var idf = query_idfs[unsafe_offset=query_index]
            if variant == 0:
                score += (
                    idf * frequency * (k1 + 1.0)
                    / (frequency + k1 * norm)
                )
            elif variant == 1:
                var normalized_frequency = frequency / norm
                score += (
                    idf * frequency * (k1 + 1.0)
                    * (normalized_frequency + delta)
                    / (k1 + normalized_frequency + delta)
                )
            else:
                score += (
                    idf * (
                        delta
                        + frequency * (k1 + 1.0)
                        / (k1 * norm + frequency)
                    )
                )
        scores[unsafe_offset=document] = score
        document += 1


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
                    idf * frequency * (k1 + 1.0)
                    / (frequency + k1 * norms[unsafe_offset=document])
                )
            elif variant == 1:
                var normalized_frequency = (
                    frequency / norms[unsafe_offset=document]
                )
                scores[unsafe_offset=document] += (
                    idf * frequency * (k1 + 1.0)
                    * (normalized_frequency + delta)
                    / (k1 + normalized_frequency + delta)
                )
            else:
                scores[unsafe_offset=document] += (
                    idf * frequency * (k1 + 1.0)
                    / (k1 * norms[unsafe_offset=document] + frequency)
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
