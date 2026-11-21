"""Retrieval evaluation: how often does the search bring back passages from the right condition?

Run with `make eval-retrieval` (needs the local database; no LLM calls, so it is free and
deterministic). Writes evals/results/retrieval.md.

Labels: each question in retrieval_cases.jsonl names the condition it is about, and a
passage counts as relevant when it comes from that condition's page. Off-topic questions
(condition null) should return nothing. The same candidates are scored at three stages:

    vector     the k nearest passages by embedding similarity alone
    reranked   the 20 nearest re-ordered by the cross-encoder, top k
    pipeline   what the chat actually sends the LLM: re-ranked, above the relevance
               threshold, at most TOP_N (it can return fewer, or none)
"""

import argparse
import json
import logging
import statistics
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from app.dependencies import get_retrieval_service
from app.domain.passages import RELEVANCE_THRESHOLD, TOP_N, Passage, select_relevant
from app.repositories.documents import DocumentRepository
from app.repositories.processed_data import ProcessedDataRepository
from evals.metrics import hit_rate, precision, recall, reciprocal_rank

HERE = Path(__file__).parent
STAGES = ("vector", "reranked", "pipeline")


@dataclass
class Case:
    question: str
    condition: str | None


def load_cases(path: Path = HERE / "retrieval_cases.jsonl") -> list[Case]:
    return [Case(**json.loads(line)) for line in path.read_text().splitlines() if line.strip()]


def stages(ranked: list[Passage], k: int, threshold: float = RELEVANCE_THRESHOLD) -> dict[str, list[Passage]]:
    by_similarity = sorted(ranked, key=lambda passage: passage.similarity, reverse=True)
    return {"vector": by_similarity[:k], "reranked": ranked[:k], "pipeline": select_relevant(ranked, threshold)}


def sweep(cases: list[Case], ranked: list[list[Passage]], thresholds: list[float]) -> list[dict]:
    """The production pipeline's trade-off at each threshold: answering real questions
    (hit rate, precision) against staying silent on off-topic ones."""
    rows = []
    for threshold in thresholds:
        on_topic = [(case, select_relevant(r, threshold)) for case, r in zip(cases, ranked) if case.condition]
        off_topic = [select_relevant(r, threshold) for case, r in zip(cases, ranked) if not case.condition]
        hits = [[p.condition == case.condition for p in passages] for case, passages in on_topic]
        rows.append({
            "threshold": threshold,
            "hit_rate": statistics.mean(hit_rate(h) for h in hits),
            "precision": statistics.mean(precision(h) for h in hits),
            "empty_on_topic": statistics.mean(not passages for _, passages in on_topic),
            "abstained_off_topic": statistics.mean(not passages for passages in off_topic) if off_topic else 0.0,
        })
    return rows


StageResults = dict[str, list[Passage]]  # what each stage returned for one question


def evaluate(cases: list[Case], ranked: list[list[Passage]], totals: dict[str, int], k: int) -> dict:
    """Score every stage on the questions about a condition, and check it stays silent on the others."""
    graded = [(case, stages(candidates, k)) for case, candidates in zip(cases, ranked, strict=True)]
    on_topic = [(case, results) for case, results in graded if case.condition is not None]
    off_topic = [results for case, results in graded if case.condition is None]
    _check_known_conditions([case for case, _ in on_topic], totals)
    return {
        "k": k,
        "on_topic": len(on_topic),
        "off_topic": len(off_topic),
        "summary": {stage: _stage_summary(stage, on_topic, off_topic, totals) for stage in STAGES},
        "misses": [_miss(case, results["pipeline"]) for case, results in on_topic if not _found(case, results["pipeline"])],
        # off-topic questions the pipeline still found passages for
        "leaks": [_leak(case, results["pipeline"]) for case, results in graded if case.condition is None and results["pipeline"]],
        "outcomes": [_outcome(case, results["pipeline"]) for case, results in graded],  # for the website's grid
    }


def _check_known_conditions(cases: list[Case], totals: dict[str, int]) -> None:
    for case in cases:
        if case.condition not in totals:
            raise SystemExit(f"Unknown condition in retrieval_cases.jsonl: {case.condition!r}")


def _hits(passages: list[Passage], condition: str | None) -> list[bool]:
    """For each passage, best first, whether it comes from the question's condition."""
    return [passage.condition == condition for passage in passages]


def _found(case: Case, passages: list[Passage]) -> bool:
    return any(_hits(passages, case.condition))


def _stage_summary(
    stage: str, on_topic: list[tuple[Case, StageResults]], off_topic: list[StageResults], totals: dict[str, int]
) -> dict[str, float]:
    """One stage's averages over the on-topic questions, and how often it stays silent."""
    hits = [(_hits(results[stage], case.condition), totals[case.condition]) for case, results in on_topic]
    return {
        "precision": statistics.mean(precision(h) for h, _ in hits),
        "recall": statistics.mean(recall(h, total) for h, total in hits),
        "hit_rate": statistics.mean(hit_rate(h) for h, _ in hits),
        "mrr": statistics.mean(reciprocal_rank(h) for h, _ in hits),
        "empty_on_topic": sum(not results[stage] for _, results in on_topic) / len(on_topic),
        "abstained_off_topic": sum(not results[stage] for results in off_topic) / len(off_topic) if off_topic else 0.0,
    }


def _miss(case: Case, pipeline: list[Passage]) -> dict:
    return {
        "question": case.question,
        "expected": case.condition,
        "got": [passage.condition for passage in pipeline] or ["(nothing above the threshold)"],
    }


def _leak(case: Case, pipeline: list[Passage]) -> dict:
    return {"question": case.question, "got": sorted({str(passage.condition) for passage in pipeline})}


def _outcome(case: Case, pipeline: list[Passage]) -> dict:
    """found / wrong / none for a question about a condition; silent / leak for the others."""
    if case.condition is None:
        outcome = "leak" if pipeline else "silent"
    else:
        outcome = "found" if _found(case, pipeline) else "wrong" if pipeline else "none"
    return {"question": case.question, "condition": case.condition, "outcome": outcome}


def report(result: dict, sweep_rows: list[dict]) -> str:
    sections = [
        _stages_section(result),
        _sweep_section(sweep_rows),
        _misses_section(result["misses"]),
        _leaks_section(result["leaks"]),
    ]
    lines = [line for section in sections for line in [*section, ""]]
    return "\n".join(lines) + "\n"


def _stages_section(result: dict) -> list[str]:
    k = result["k"]
    labels = {"vector": f"Vector search, top {k}", "reranked": f"+ cross-encoder re-rank, top {k}", "pipeline": "+ relevance threshold (production)"}
    rows = []
    for stage in STAGES:
        s = result["summary"][stage]
        rows.append(
            f"| {labels[stage]} | {s['precision']:.0%} | {s['recall']:.0%} | {s['hit_rate']:.0%} | {s['mrr']:.2f} "
            f"| {s['empty_on_topic']:.0%} | {s['abstained_off_topic']:.0%} |"
        )
    return [
        f"# Retrieval evaluation ({date.today().isoformat()})",
        "",
        f"Knowledge base: {result['documents']} passages from worker {result['processor']}.",
        "",
        (
            f"{result['on_topic']} questions about a known condition and {result['off_topic']} off-topic questions "
            f"(evals/retrieval_cases.jsonl). A passage is relevant when it comes from the question's condition. "
            f"k = {k}; the pipeline keeps at most {TOP_N} passages scoring above "
            f"{RELEVANCE_THRESHOLD} on the cross-encoder."
        ),
        "",
        "| Stage | Precision | Recall | Hit rate | MRR | On-topic with no passages | Off-topic correctly empty |",
        "| --- | --- | --- | --- | --- | --- | --- |",
        *rows,
        "",
        (
            "Precision: share of returned passages from the right condition. Recall: share of that condition's "
            f"passages returned (at most {k} of its 4-14 passages, so it cannot reach 100%). Hit rate: at least one "
            "right passage. MRR: 1 / rank of the first right passage."
        ),
    ]


def _sweep_section(sweep_rows: list[dict]) -> list[str]:
    return [
        "## Relevance threshold sweep",
        "",
        (
            "The production stage at other cross-encoder thresholds: a lower threshold answers more real "
            "questions but lets more off-topic ones through."
        ),
        "",
        "| Threshold | Hit rate | Precision | On-topic with no passages | Off-topic correctly empty |",
        "| --- | --- | --- | --- | --- |",
        *(
            f"| {row['threshold']:g}{' (current)' if row['threshold'] == RELEVANCE_THRESHOLD else ''} "
            f"| {row['hit_rate']:.0%} | {row['precision']:.0%} | {row['empty_on_topic']:.0%} | {row['abstained_off_topic']:.0%} |"
            for row in sweep_rows
        ),
    ]


def _misses_section(misses: list[dict]) -> list[str]:
    table = [
        "| Question | Expected | Conditions returned |",
        "| --- | --- | --- |",
        *(f"| {m['question']} | {m['expected']} | {', '.join(str(c) for c in m['got'])} |" for m in misses),
    ]
    return [f"## Pipeline misses ({len(misses)})", "", *(table if misses else ["None."])]


def _leaks_section(leaks: list[dict]) -> list[str]:
    return [
        f"## Off-topic questions that still got passages ({len(leaks)})",
        "",
        *([f"- {leak['question']} → {', '.join(leak['got'])}" for leak in leaks] or ["None."]),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--k", type=int, default=5, help="results scored per question for the vector and reranked stages")
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING)

    cases = load_cases()
    retrieval_service = get_retrieval_service()
    ranked = [retrieval_service.rank(case.question) for case in cases]  # the slow part: done once
    totals = DocumentRepository().passage_counts_by_condition()
    result = evaluate(cases, ranked, totals, args.k)
    processor = ", ".join(ProcessedDataRepository().processor_versions()) or "none"
    result |= {"documents": sum(totals.values()), "processor": processor}
    thresholds = [-6.0, -5.0, -4.0, -3.0, -2.0, -1.0, 0.0]
    sweep_rows = sweep(cases, ranked, thresholds)
    text = report(result, sweep_rows)
    out = HERE / "results" / "retrieval.md"
    out.parent.mkdir(exist_ok=True)
    out.write_text(text)
    # The same numbers for the website (medical_conditions_react_frontend/src/data/evaluation/retrieval.json).
    data = {
        "date": date.today().isoformat(),
        "processor": result["processor"],
        "documents": result["documents"],
        "k": result["k"],
        "topN": TOP_N,
        "threshold": RELEVANCE_THRESHOLD,
        "onTopic": result["on_topic"],
        "offTopic": result["off_topic"],
        "stages": result["summary"],
        "sweep": sweep_rows,
        "misses": result["misses"],
        "leaks": result["leaks"],
        # found / wrong / none for questions about a condition; silent / leak for the others
        "outcomes": result["outcomes"],
    }
    (HERE / "results" / "retrieval.json").write_text(json.dumps(data, indent=2) + "\n")
    print(text)
    print(f"Written to {out.relative_to(HERE.parent)}")


if __name__ == "__main__":
    main()
