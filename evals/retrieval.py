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

from sqlalchemy import func, select

from app.core.database import SessionLocal
from app.core.models import Document, ProcessedData
from app.services import retrieval
from app.services.retrieval import Passage
from evals.metrics import hit_rate, precision, recall, reciprocal_rank

HERE = Path(__file__).parent
STAGES = ("vector", "reranked", "pipeline")


@dataclass
class Case:
    question: str
    condition: str | None


def load_cases(path: Path = HERE / "retrieval_cases.jsonl") -> list[Case]:
    return [Case(**json.loads(line)) for line in path.read_text().splitlines() if line.strip()]


def passages_per_condition() -> dict[str, int]:
    with SessionLocal() as session:
        query = (
            select(ProcessedData.condition_name, func.count())
            .join(Document, Document.processed_data_id == ProcessedData.id)
            .group_by(ProcessedData.condition_name)
        )
        return dict(session.execute(query).all())


def processor_versions() -> str:
    """The worker version(s) that produced the passages being scored."""
    with SessionLocal() as session:
        versions = session.scalars(select(ProcessedData.processor_version).distinct().order_by(ProcessedData.processor_version))
        return ", ".join(versions) or "none"


def stages(ranked: list[Passage], k: int, threshold: float = retrieval.RELEVANCE_THRESHOLD) -> dict[str, list[Passage]]:
    by_similarity = sorted(ranked, key=lambda passage: passage.similarity, reverse=True)
    return {"vector": by_similarity[:k], "reranked": ranked[:k], "pipeline": pipeline(ranked, threshold)}


def pipeline(ranked: list[Passage], threshold: float) -> list[Passage]:
    """What retrieval.search would return with this threshold."""
    return [passage for passage in ranked if passage.relevance > threshold][: retrieval.TOP_N]


def sweep(cases: list[Case], ranked: list[list[Passage]], thresholds: list[float]) -> list[dict]:
    """The production pipeline's trade-off at each threshold: answering real questions
    (hit rate, precision) against staying silent on off-topic ones."""
    rows = []
    for threshold in thresholds:
        on_topic = [(case, pipeline(r, threshold)) for case, r in zip(cases, ranked) if case.condition]
        off_topic = [pipeline(r, threshold) for case, r in zip(cases, ranked) if not case.condition]
        hits = [[p.condition == case.condition for p in passages] for case, passages in on_topic]
        rows.append({
            "threshold": threshold,
            "hit_rate": statistics.mean(hit_rate(h) for h in hits),
            "precision": statistics.mean(precision(h) for h in hits),
            "empty_on_topic": statistics.mean(not passages for _, passages in on_topic),
            "abstained_off_topic": statistics.mean(not passages for passages in off_topic) if off_topic else 0.0,
        })
    return rows


def evaluate(cases: list[Case], ranked: list[list[Passage]], totals: dict[str, int], k: int) -> dict:
    scores = {stage: {"precision": [], "recall": [], "hit_rate": [], "mrr": [], "empty_on_topic": 0} for stage in STAGES}
    off_topic_empty = {stage: 0 for stage in STAGES}
    misses = []
    leaks = []  # off-topic questions the pipeline still found passages for
    outcomes = []  # one per question, for the website's grid

    for case, candidates in zip(cases, ranked, strict=True):
        results = stages(candidates, k)
        if case.condition is None:
            for stage, passages in results.items():
                off_topic_empty[stage] += not passages
            if results["pipeline"]:
                leaks.append({"question": case.question, "got": sorted({str(p.condition) for p in results["pipeline"]})})
            outcomes.append({"question": case.question, "condition": None, "outcome": "leak" if results["pipeline"] else "silent"})
            continue
        if case.condition not in totals:
            raise SystemExit(f"Unknown condition in retrieval_cases.jsonl: {case.condition!r}")
        for stage, passages in results.items():
            hits = [passage.condition == case.condition for passage in passages]
            stage_scores = scores[stage]
            stage_scores["precision"].append(precision(hits))
            stage_scores["recall"].append(recall(hits, totals[case.condition]))
            stage_scores["hit_rate"].append(hit_rate(hits))
            stage_scores["mrr"].append(reciprocal_rank(hits))
            stage_scores["empty_on_topic"] += not passages
        pipeline = results["pipeline"]
        found = any(passage.condition == case.condition for passage in pipeline)
        outcomes.append({
            "question": case.question,
            "condition": case.condition,
            "outcome": "found" if found else "wrong" if pipeline else "none",
        })
        if not found:
            misses.append({
                "question": case.question,
                "expected": case.condition,
                "got": [passage.condition for passage in pipeline] or ["(nothing above the threshold)"],
            })

    on_topic = sum(case.condition is not None for case in cases)
    off_topic = len(cases) - on_topic
    summary = {
        stage: {
            **{name: statistics.mean(values) for name, values in metrics.items() if isinstance(values, list)},
            "empty_on_topic": metrics["empty_on_topic"] / on_topic,
            "abstained_off_topic": off_topic_empty[stage] / off_topic if off_topic else 0.0,
        }
        for stage, metrics in scores.items()
    }
    return {"k": k, "on_topic": on_topic, "off_topic": off_topic, "summary": summary, "misses": misses, "leaks": leaks, "outcomes": outcomes}


def report(result: dict, sweep_rows: list[dict]) -> str:
    k = result["k"]
    lines = [
        f"# Retrieval evaluation ({date.today().isoformat()})",
        "",
        f"Knowledge base: {result['documents']} passages from worker {result['processor']}.",
        "",
        (
            f"{result['on_topic']} questions about a known condition and {result['off_topic']} off-topic questions "
            f"(evals/retrieval_cases.jsonl). A passage is relevant when it comes from the question's condition. "
            f"k = {k}; the pipeline keeps at most {retrieval.TOP_N} passages scoring above "
            f"{retrieval.RELEVANCE_THRESHOLD} on the cross-encoder."
        ),
        "",
        "| Stage | Precision | Recall | Hit rate | MRR | On-topic with no passages | Off-topic correctly empty |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    labels = {"vector": f"Vector search, top {k}", "reranked": f"+ cross-encoder re-rank, top {k}", "pipeline": "+ relevance threshold (production)"}
    for stage in STAGES:
        s = result["summary"][stage]
        lines.append(
            f"| {labels[stage]} | {s['precision']:.0%} | {s['recall']:.0%} | {s['hit_rate']:.0%} | {s['mrr']:.2f} "
            f"| {s['empty_on_topic']:.0%} | {s['abstained_off_topic']:.0%} |"
        )
    lines += [
        "",
        (
            "Precision: share of returned passages from the right condition. Recall: share of that condition's "
            f"passages returned (at most {k} of its 4-14 passages, so it cannot reach 100%). Hit rate: at least one "
            "right passage. MRR: 1 / rank of the first right passage."
        ),
        "",
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
            f"| {row['threshold']:g}{' (current)' if row['threshold'] == retrieval.RELEVANCE_THRESHOLD else ''} "
            f"| {row['hit_rate']:.0%} | {row['precision']:.0%} | {row['empty_on_topic']:.0%} | {row['abstained_off_topic']:.0%} |"
            for row in sweep_rows
        ),
        "",
        f"## Pipeline misses ({len(result['misses'])})",
        "",
    ]
    lines_leaks = [
        f"## Off-topic questions that still got passages ({len(result['leaks'])})",
        "",
        *([f"- {leak['question']} → {', '.join(leak['got'])}" for leak in result["leaks"]] or ["None."]),
        "",
    ]
    if result["misses"]:
        lines += ["| Question | Expected | Conditions returned |", "| --- | --- | --- |"]
        lines += [f"| {m['question']} | {m['expected']} | {', '.join(str(c) for c in m['got'])} |" for m in result["misses"]]
    else:
        lines.append("None.")
    lines += ["", *lines_leaks]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--k", type=int, default=5, help="results scored per question for the vector and reranked stages")
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING)

    cases = load_cases()
    ranked = [retrieval.rank(case.question) for case in cases]  # the slow part: done once
    totals = passages_per_condition()
    result = evaluate(cases, ranked, totals, args.k)
    result |= {"documents": sum(totals.values()), "processor": processor_versions()}
    thresholds = [-6.0, -5.0, -4.0, -3.0, -2.0, -1.0, 0.0]
    sweep_rows = sweep(cases, ranked, thresholds)
    text = report(result, sweep_rows)
    out = HERE / "results" / "retrieval.md"
    out.parent.mkdir(exist_ok=True)
    out.write_text(text)
    # The same numbers for the website (medical_conditions_react_frontend/src/data/evaluation.json).
    data = {
        "date": date.today().isoformat(),
        "processor": result["processor"],
        "documents": result["documents"],
        "k": result["k"],
        "topN": retrieval.TOP_N,
        "threshold": retrieval.RELEVANCE_THRESHOLD,
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
