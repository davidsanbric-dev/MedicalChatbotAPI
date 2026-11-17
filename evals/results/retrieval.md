# Retrieval evaluation (2026-11-17)

Knowledge base: 565 passages from worker v6.

76 questions about a known condition and 16 off-topic questions (evals/retrieval_cases.jsonl). A passage is relevant when it comes from the question's condition. k = 5; the pipeline keeps at most 5 passages scoring above -3.0 on the cross-encoder.

| Stage | Precision | Recall | Hit rate | MRR | On-topic with no passages | Off-topic correctly empty |
| --- | --- | --- | --- | --- | --- | --- |
| Vector search, top 5 | 66% | 46% | 100% | 0.95 | 0% | 0% |
| + cross-encoder re-rank, top 5 | 63% | 44% | 100% | 0.93 | 0% | 0% |
| + relevance threshold (production) | 71% | 37% | 92% | 0.88 | 5% | 94% |

Precision: share of returned passages from the right condition. Recall: share of that condition's passages returned (at most 5 of its 4-14 passages, so it cannot reach 100%). Hit rate: at least one right passage. MRR: 1 / rank of the first right passage.

## Relevance threshold sweep

The production stage at other cross-encoder thresholds: a lower threshold answers more real questions but lets more off-topic ones through.

| Threshold | Hit rate | Precision | On-topic with no passages | Off-topic correctly empty |
| --- | --- | --- | --- | --- |
| -6 | 99% | 67% | 0% | 88% |
| -5 | 99% | 70% | 0% | 88% |
| -4 | 93% | 69% | 4% | 94% |
| -3 (current) | 92% | 71% | 5% | 94% |
| -2 | 89% | 72% | 8% | 94% |
| -1 | 88% | 77% | 11% | 94% |
| 0 | 86% | 77% | 12% | 100% |

## Pipeline misses (6)

| Question | Expected | Conditions returned |
| --- | --- | --- |
| How long does a chest cold with a cough last? | Bronchitis | (nothing above the threshold) |
| Why do my fingers go numb and tingle at night? | Carpal tunnel syndrome | Panic attack |
| What causes an itchy red rash after touching a new soap? | Dermatitis | (nothing above the threshold) |
| What are the early signs of high blood sugar? | Diabetes | (nothing above the threshold) |
| What helps when I keep throwing up? | Nausea and vomiting | (nothing above the threshold) |
| How long does the stomach flu last? | Stomach flu | Diarrhea, Diarrhea |

## Off-topic questions that still got passages (1)

- What are the symptoms of appendicitis? → Bursitis, Cellulitis

