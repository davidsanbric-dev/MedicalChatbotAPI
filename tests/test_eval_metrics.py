from app.services.retrieval import Passage
from evals.metrics import hit_rate, precision, recall, reciprocal_rank
from evals.retrieval import pipeline, stages


def test_ranking_metrics():
    hits = [False, True, True, False]
    assert precision(hits) == 0.5
    assert recall(hits, total_relevant=4) == 0.5
    assert hit_rate(hits) == 1.0
    assert reciprocal_rank(hits) == 0.5


def test_metrics_of_an_empty_result():
    assert precision([]) == recall([], 3) == hit_rate([]) == reciprocal_rank([]) == 0.0


def test_stages_order_by_similarity_then_relevance_then_threshold():
    ranked = [  # already in cross-encoder order, as retrieval.rank returns them
        Passage("a", similarity=0.5, relevance=3.0, condition="Acne"),
        Passage("b", similarity=0.9, relevance=-1.0, condition="Rash"),
        Passage("c", similarity=0.7, relevance=-4.0, condition="Acne"),
    ]
    result = stages(ranked, k=2, threshold=-2.0)
    assert [p.content for p in result["vector"]] == ["b", "c"]
    assert [p.content for p in result["reranked"]] == ["a", "b"]
    assert [p.content for p in result["pipeline"]] == ["a", "b"]
    assert [p.content for p in pipeline(ranked, threshold=0.0)] == ["a"]
