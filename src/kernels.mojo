"""BM25 scoring kernels exported through a small C ABI."""

from std.algorithm import parallelize
from std.runtime import initialize_runtime
from std.sys.info import simd_width_of

comptime FPtr = UnsafePointer[Float64, AnyOrigin[mut=True]]
comptime IPtr = UnsafePointer[Int64, AnyOrigin[mut=True]]
comptime DENSE_PARALLEL_THRESHOLD = 16_384
comptime DENSE_BLOCK_SIZE = 4_096


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
        var length = document_lengths.load[width=W](document)
        var norm = 1.0 - b + b * length / average_length
        var score = SIMD[DType.float64, W](0.0)
        for query_index in range(query_count):
            var frequency = frequencies.load[width=W](
                query_index * document_count + document
            )
            var idf = query_idfs[query_index]
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
        scores.store(document, score)
        document += W

    while document < end:
        var norm = (
            1.0 - b + b * document_lengths[document] / average_length
        )
        var score = 0.0
        for query_index in range(query_count):
            var frequency = frequencies[
                query_index * document_count + document
            ]
            var idf = query_idfs[query_index]
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
        scores[document] = score
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
            base += query_idfs[query_index] * delta
    for document in range(document_count):
        scores[document] = base

    for query_index in range(query_count):
        var term = Int(query_terms[query_index])
        var idf = query_idfs[query_index]
        for posting in range(Int(offsets[term]), Int(offsets[term + 1])):
            var document = Int(docs[posting])
            var frequency = frequencies[posting]
            if variant == 0:
                scores[document] += (
                    idf * frequency * (k1 + 1.0)
                    / (frequency + k1 * norms[document])
                )
            elif variant == 1:
                var normalized_frequency = frequency / norms[document]
                scores[document] += (
                    idf * frequency * (k1 + 1.0)
                    * (normalized_frequency + delta)
                    / (k1 + normalized_frequency + delta)
                )
            else:
                scores[document] += (
                    idf * frequency * (k1 + 1.0)
                    / (k1 * norms[document] + frequency)
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

    if document_count < DENSE_PARALLEL_THRESHOLD:
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
    else:
        initialize_runtime()
        var block_count = (
            document_count + DENSE_BLOCK_SIZE - 1
        ) // DENSE_BLOCK_SIZE

        @parameter
        def score_block(block: Int):
            var start = block * DENSE_BLOCK_SIZE
            var end = min(start + DENSE_BLOCK_SIZE, document_count)
            score_dense_range(
                frequencies,
                query_idfs,
                document_lengths,
                scores,
                start,
                end,
                document_count,
                query_count,
                variant,
                k1,
                b,
                delta,
                average_length,
            )

        parallelize[score_block](block_count)
