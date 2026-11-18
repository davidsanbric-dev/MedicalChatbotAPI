"""The domain rules on their own: no database, model or framework."""

import pytest

from app.domain.handoff import Handoff, Urgency
from app.domain.passages import TOP_N, Passage, select_relevant
from app.domain.symptoms import most_shared
from app.domain.triage import is_emergency


@pytest.mark.parametrize(
    "message",
    ["I've been thinking about ending my life", "I want to kill myself", "Mi hijo está inconsciente", "No puedo respirar"],
)
def test_emergency_wording_is_recognized(message):
    assert is_emergency(message)


@pytest.mark.parametrize("message", ["My chest feels congested from a cold", "How long does the flu last?"])
def test_ordinary_questions_are_not_emergencies(message):
    assert not is_emergency(message)


def test_only_relevant_passages_are_selected_best_first():
    ranked = [Passage("a", 0.5, 3.0), Passage("b", 0.9, -1.0), Passage("c", 0.7, -4.0)]
    assert [p.content for p in select_relevant(ranked)] == ["a", "b"]
    assert [p.content for p in select_relevant(ranked, threshold=0.0)] == ["a"]
    assert select_relevant(ranked, threshold=3.0) == []  # the threshold itself is not relevant


def test_at_most_top_n_passages_are_selected():
    ranked = [Passage(str(i), 0.5, 1.0) for i in range(TOP_N + 3)]
    assert len(select_relevant(ranked)) == TOP_N


def test_a_handoff_is_identified_by_its_id():
    first = Handoff("s", Urgency.ROUTINE, "Wants a refill.", "refill please")
    same_id = Handoff("other", Urgency.URGENT, "Changed.", "...", id=first.id)
    assert first == same_id
    assert first != Handoff("s", Urgency.ROUTINE, "Wants a refill.", "refill please")


def test_the_emergency_handoff_is_marked_emergency():
    handoff = Handoff.for_emergency("s", "chest pain")
    assert handoff.urgency is Urgency.EMERGENCY and handoff.last_message == "chest pain"


def test_symptoms_are_ranked_by_how_many_conditions_list_them():
    blocks = [
        ["Symptoms", ["Fever", " Cough ", "Fever"]],  # repeated within one condition: counted once
        ["Symptoms", ["Cough", "Rash", ""]],
        ["Symptoms", "not a list"],  # malformed records are skipped
        None,
    ]
    assert most_shared(blocks, limit=10) == [("Cough", 2), ("Fever", 1), ("Rash", 1)]
    assert most_shared(blocks, limit=1) == [("Cough", 2)]
