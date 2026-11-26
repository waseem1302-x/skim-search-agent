"""The public ranking bound with a scripted model and an in-memory code executor."""
from copy import copy

import pytest

from agent_search import research
from agent_search.api import as_units, build_agent
from agent_search.strategies.base import STRATEGIES
from agent_search.strategies.conditions import CONDITIONS, Condition
from agent_search.tasks.base import TASKS


QUESTION = "Which document contains the gold answer?"
DOCS = [
    {"_id": "z_gold", "title": "Gold", "text": "goldneedle identifies the answer."},
    {"_id": "a_second", "title": "Second", "text": "secondneedle identifies another document."},
    {"_id": "m_third", "title": "Third", "text": "thirdneedle identifies a third document."},
    {"_id": "b_fourth", "title": "Fourth", "text": "fourthneedle identifies a fourth document."},
]
FULL_RANKING = ["z_gold", "a_second", "m_third", "b_fourth"]


@pytest.fixture
def offline_strategy(monkeypatch):
    task = copy(TASKS["research"])
    task.domain = "code"
    name = "__test_api_k_offline"
    monkeypatch.setitem(CONDITIONS, name, Condition(name, task, STRATEGIES["sieve_bm25"]))
    return name


def scripted_model():
    replies = iter([
        '<tool_call>{"name": "search_s", "arguments": {"query": "goldneedle"}}</tool_call>',
        '<tool_call>{"name": "search_s", "arguments": {"query": "secondneedle"}}</tool_call>',
        '<tool_call>{"name": "search_s", "arguments": {"query": "thirdneedle"}}</tool_call>',
        '<tool_call>{"name": "search_s", "arguments": {"query": "fourthneedle"}}</tool_call>',
        '<tool_call>{"name": "fetch_s", "arguments": {"specs": [[1, "(intro)"]]}}</tool_call>',
    ])

    def generate(messages):
        return next(replies, "Exact Answer: Gold")

    return generate


def run_research(strategy, **kwargs):
    return research(QUESTION, DOCS, strategy=strategy, generate=scripted_model(), max_steps=8, **kwargs)


def test_research_k_one_returns_only_the_gold_document(offline_strategy):
    result = run_research(offline_strategy, k=1)
    assert len(result.ranking) == 1
    assert result.ranking == ["z_gold"]


def test_research_k_three_returns_the_first_three_documents(offline_strategy):
    result = run_research(offline_strategy, k=3)
    assert len(result.ranking) <= 3
    assert result.ranking == FULL_RANKING[:3]


def test_research_default_k_returns_a_smaller_full_ranking(offline_strategy):
    result = run_research(offline_strategy)
    assert result.answer == "Gold"
    assert [step["action"] for step in result.steps] == ["search_s"] * 4 + ["fetch_s", "answer"]
    assert result.ranking == FULL_RANKING
    assert len(result.ranking) < 10


def test_research_none_returns_the_full_ranking(offline_strategy):
    result = run_research(offline_strategy, k=None)
    assert result.ranking == FULL_RANKING
    assert set(result.surfaced) == set(FULL_RANKING)


def test_agent_search_remains_unbounded_while_research_is_bounded(offline_strategy):
    agent = build_agent(offline_strategy, generate=scripted_model(), max_steps=8)
    agent.index(as_units(DOCS))
    assert agent.search(QUESTION, k=2) == FULL_RANKING
    result = run_research(offline_strategy, k=2)
    assert len(result.ranking) <= 2
    assert result.ranking == FULL_RANKING[:2]


@pytest.mark.parametrize("k", [0, 2, 4, 10])
def test_research_ranking_preserves_surfaced_order(offline_strategy, k):
    result = run_research(offline_strategy, k=k)
    assert result.ranking == FULL_RANKING[:k]
    assert set(result.surfaced) == set(FULL_RANKING)
