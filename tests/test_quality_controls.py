"""Quality controls fail closed and preserve the exact evidence contract."""
from types import SimpleNamespace

import pytest

from finsight.config import Settings
from finsight.graph.builder import build_agent
from finsight.guardrails.claims import assess_claim_support
from finsight.rag.format import render_context
from finsight.rag.models import Chunk, RetrievedChunk
from finsight.rag.rerank import rerank


def hit(doc_id, text, rank=1):
    return RetrievedChunk(Chunk(f'{doc_id}:0', doc_id, doc_id, text, 0), 0.1, rank)


class Model:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.prompts = []

    def with_structured_output(self, schema):
        raise NotImplementedError

    def invoke(self, prompt):
        self.prompts.append(prompt)
        return SimpleNamespace(content=next(self.responses))


class Retriever:
    def __init__(self, chunks):
        self.chunks = chunks
        self.requests = []

    def retrieve(self, query, top_k=None, **kwargs):
        self.requests.append((query, top_k, kwargs))
        return self.chunks[:top_k]


def test_context_respects_injected_tokenizer_and_preserves_reference_numbers():
    chunks = [hit('a', 'one two three four five'), hit('b', 'six')]
    rendered = render_context(chunks, max_tokens=3, token_counter=lambda text: len(text.split()))
    assert list(rendered.citations) == [2]
    assert len(rendered.text.split()) <= 3


def test_unicode_context_uses_conservative_utf8_budget_by_default():
    rendered = render_context([hit('a', '世界')], max_tokens=11)
    assert rendered.citations == {}  # 14 UTF-8 bytes with its reference header
    assert len(rendered.text.encode()) <= 11


@pytest.mark.parametrize('verdict', [
    '{"segments":[]}',
    '{"segments":[{"segment":2,"supported":true}]}',
    '{"segments":[{"segment":1,"supported":"true"}]}',
    '{"segments":[{"segment":1,"supported":true},{"segment":1,"supported":true}]}',
])
def test_semantic_support_rejects_missing_extra_or_coerced_verdicts(verdict):
    model = Model([verdict, verdict])
    result = assess_claim_support(model, 'The amount is 50 [1].', {1: hit('a', 'Amount: 50.')})
    assert not result.supported


def test_semantic_judge_sees_only_sources_cited_by_that_segment():
    model = Model(['{"segments":[{"segment":1,"supported":false}]}'])
    result = assess_claim_support(model, 'The amount is 65 [1].', {
        1: hit('a', 'The amount is 50.'), 2: hit('b', 'Secret unrelated text: 65.')
    })
    assert not result.supported
    assert 'Secret unrelated text' not in model.prompts[0]
    assert result.unsupported_segments == (1,)


def test_uncited_sentence_cannot_hide_behind_valid_citation_elsewhere():
    model = Model([])
    result = assess_claim_support(model, 'An unsupported statement. Amount: 50 [1].', {
        1: hit('a', 'Amount: 50.')
    })
    assert not result.supported
    assert model.prompts == []


def test_semantic_failure_abstains_when_graph_budget_exhausted():
    model = Model([
        '{"sufficient":true}', 'The amount is 65 [1].',
        '{"segments":[{"segment":1,"supported":false}]}',
    ])
    agent = build_agent(Settings(max_retrieval_attempts=0, semantic_verification=True),
                        retriever=Retriever([hit('a', 'The amount is 50.')]), llm=model)
    final = agent.invoke({'question':'What is the amount?'})
    assert final['semantic_supported'] is False
    assert 'I cannot provide' in final['answer']
    assert '65' not in final['answer']
    assert not final['grounded']
    assert final['runtime']['model_calls'] == 3


def test_valid_semantic_answer_is_retained():
    model = Model([
        '{"sufficient":true}', 'The amount is 50 [1].',
        '{"segments":[{"segment":1,"supported":true}]}',
    ])
    final = build_agent(Settings(max_retrieval_attempts=0, semantic_verification=True),
                        retriever=Retriever([hit('a', 'The amount is 50.')]), llm=model
                        ).invoke({'question':'What is the amount?'})
    assert final['grounded'] and final['semantic_supported']
    assert final['answer'] == 'The amount is 50 [1].'


def test_reranker_reorders_real_chunks_and_preserves_original_scores():
    chunks = [hit('a', 'one'), hit('b', 'two', 2)]
    result = rerank(Model(['{"references":[2,1]}']), 'two', chunks, top_k=1, max_chars=1000)
    assert result.applied
    assert result.chunks[0].chunk.doc_id == 'b'
    assert result.chunks[0].rank == 1 and result.chunks[0].score == 0.1


def test_reranker_rejects_invented_reference():
    chunks = [hit('a', 'one'), hit('b', 'two', 2)]
    result = rerank(Model(['{"references":[9,1]}']), 'two', chunks, top_k=1, max_chars=1000)
    assert not result.applied and result.chunks == chunks[:1]


def test_graph_fetches_larger_pool_then_truncates_reranked_candidates():
    retriever = Retriever([hit('a', 'one'), hit('b', 'two', 2)])
    model = Model(['{"references":[2,1]}', '{"sufficient":true}', 'Two [1].'])
    settings = Settings(rerank_enabled=True, retrieval_top_k=1, retrieval_candidates=5,
                        max_retrieval_attempts=0)
    final = build_agent(settings, retriever=retriever, llm=model).invoke({'question':'two?'})
    assert retriever.requests[0][1] == 5
    assert final['rerank_applied']
    assert [item.chunk.doc_id for item in final['retrieved']] == ['b']
    assert final['context_citations'][1].chunk.doc_id == 'b'
    assert final['runtime']['model_calls'] == 3


def test_semantic_evidence_preserves_displayed_publication_date():
    chunk = RetrievedChunk(Chunk(
        'a:0', 'a', 'a', 'Revenue rose.', 0,
        source_url='https://example.test/report', published_at='2024-02-28',
        retrieved_at='2026-10-03', revision='sha256:example',
    ), 0.1, 1)
    model = Model(['{"segments":[{"segment":1,"supported":true}]}'])
    result = assess_claim_support(model, 'Published 2024-02-28 [1].', {1: chunk})
    assert result.supported
    assert render_context([chunk]).text in model.prompts[0]
