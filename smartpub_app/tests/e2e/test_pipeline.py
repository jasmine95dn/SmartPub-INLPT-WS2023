"""
E2E tests for pipeline() — tests orchestration across all layers with
Pinecone and HuggingFace replaced by mocks.
"""
import pytest
from unittest.mock import patch, MagicMock
from model.model import pipeline


def _setup_rag_mock(answer: str):
    """Return a mock create_retrieval_chain that yields *answer* on invoke."""
    mock_chain = MagicMock()
    mock_chain.return_value.invoke.return_value = {"answer": answer}
    return mock_chain


# ── pipeline() return value ───────────────────────────────────────────────────

class TestPipelineReturn:
    def test_returns_rag_answer(self):
        mock_chain = _setup_rag_mock("COVID-19 causes pneumonia.")
        with patch("model.model.DBRetriever"), \
             patch("model.model.QA"), \
             patch("model.model.create_stuff_documents_chain"), \
             patch("model.model.create_retrieval_chain", mock_chain):
            result = pipeline(api_key="key", question="What does COVID cause?", hf_auth="token")
        assert result == "COVID-19 causes pneumonia."

    def test_invoke_called_with_question_as_input(self):
        mock_chain = _setup_rag_mock("answer")
        with patch("model.model.DBRetriever"), \
             patch("model.model.QA"), \
             patch("model.model.create_stuff_documents_chain"), \
             patch("model.model.create_retrieval_chain", mock_chain):
            pipeline(api_key="k", question="What is aspirin?", hf_auth="t")
        mock_chain.return_value.invoke.assert_called_once_with({"input": "What is aspirin?"})


# ── DBRetriever construction ──────────────────────────────────────────────────

class TestPipelineRetrieverConstruction:
    def test_passes_api_key_to_retriever(self):
        mock_chain = _setup_rag_mock("ans")
        with patch("model.model.DBRetriever") as MockDB, \
             patch("model.model.QA"), \
             patch("model.model.create_stuff_documents_chain"), \
             patch("model.model.create_retrieval_chain", mock_chain):
            pipeline(api_key="my-pinecone-key", question="Q?", hf_auth="token")
        _, kwargs = MockDB.call_args
        assert kwargs["api_key"] == "my-pinecone-key"

    def test_passes_hf_auth_to_retriever(self):
        mock_chain = _setup_rag_mock("ans")
        with patch("model.model.DBRetriever") as MockDB, \
             patch("model.model.QA"), \
             patch("model.model.create_stuff_documents_chain"), \
             patch("model.model.create_retrieval_chain", mock_chain):
            pipeline(api_key="k", question="Q?", hf_auth="my-hf-token")
        _, kwargs = MockDB.call_args
        assert kwargs["hf_auth"] == "my-hf-token"

    def test_passes_index_name_to_retriever(self):
        mock_chain = _setup_rag_mock("ans")
        with patch("model.model.DBRetriever") as MockDB, \
             patch("model.model.QA"), \
             patch("model.model.create_stuff_documents_chain"), \
             patch("model.model.create_retrieval_chain", mock_chain):
            pipeline(api_key="k", question="Q?", hf_auth="t", index_name="my-index")
        _, kwargs = MockDB.call_args
        assert kwargs["index_name"] == "my-index"


# ── QA construction ───────────────────────────────────────────────────────────

class TestPipelineQAConstruction:
    def test_passes_question_as_prompt(self):
        mock_chain = _setup_rag_mock("ans")
        with patch("model.model.DBRetriever"), \
             patch("model.model.QA") as MockQA, \
             patch("model.model.create_stuff_documents_chain"), \
             patch("model.model.create_retrieval_chain", mock_chain):
            pipeline(api_key="k", question="What is aspirin?", hf_auth="t")
        _, kwargs = MockQA.call_args
        assert kwargs["prompt"] == "What is aspirin?"

    def test_passes_hf_auth_to_qa(self):
        mock_chain = _setup_rag_mock("ans")
        with patch("model.model.DBRetriever"), \
             patch("model.model.QA") as MockQA, \
             patch("model.model.create_stuff_documents_chain"), \
             patch("model.model.create_retrieval_chain", mock_chain):
            pipeline(api_key="k", question="Q?", hf_auth="hf-secret")
        _, kwargs = MockQA.call_args
        assert kwargs["hf_auth"] == "hf-secret"


# ── RAG chain construction ────────────────────────────────────────────────────

class TestPipelineRAGConstruction:
    def test_stuff_documents_chain_is_built(self):
        mock_chain = _setup_rag_mock("ans")
        with patch("model.model.DBRetriever"), \
             patch("model.model.QA") as MockQA, \
             patch("model.model.create_stuff_documents_chain") as mock_stuff, \
             patch("model.model.create_retrieval_chain", mock_chain):
            pipeline(api_key="k", question="Q?", hf_auth="t")
        mock_stuff.assert_called_once()
        args, _ = mock_stuff.call_args
        assert args[0] is MockQA.return_value.llm

    def test_k_controls_retriever_search_kwargs(self):
        mock_chain = _setup_rag_mock("ans")
        mock_db = MagicMock()
        with patch("model.model.DBRetriever", return_value=mock_db), \
             patch("model.model.QA"), \
             patch("model.model.create_stuff_documents_chain"), \
             patch("model.model.create_retrieval_chain", mock_chain):
            pipeline(api_key="k", question="Q?", hf_auth="t", k=5)
        mock_db.vectorstore_db.as_retriever.assert_called_once_with(
            search_kwargs={"k": 5}
        )

    def test_default_k_is_10(self):
        mock_chain = _setup_rag_mock("ans")
        mock_db = MagicMock()
        with patch("model.model.DBRetriever", return_value=mock_db), \
             patch("model.model.QA"), \
             patch("model.model.create_stuff_documents_chain"), \
             patch("model.model.create_retrieval_chain", mock_chain):
            pipeline(api_key="k", question="Q?", hf_auth="t")
        mock_db.vectorstore_db.as_retriever.assert_called_once_with(
            search_kwargs={"k": 10}
        )
