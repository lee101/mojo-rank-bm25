from rank_bm25 import BM25Okapi

documents = [
    "the quick brown fox".split(),
    "the slow brown bear".split(),
    "a fox is quick".split(),
    "a turtle moves slowly".split(),
    "birds fly over water".split(),
]

index = BM25Okapi(documents)
query = "quick fox".split()
print(index.get_scores(query))
print(index.get_top_n(query, documents, n=2))
