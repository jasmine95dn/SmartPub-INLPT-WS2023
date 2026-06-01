"""
E2E tests for pipeline() — tests orchestration across all layers with
Pinecone and HuggingFace replaced by mocks.
"""
import pytest
from unittest.mock import patch, MagicMock, call
from model.model import pipeline


def _rag_mock(answer: str) -> MagicMock:
    """Build a RetrievalQA mock whose ['result'] returns *answer*."""
    instance = MagicMock()
    instance.__getitem__ = MagicMock(return_value=answer)
    return instance


# ── pipeline() return value ───────────────────────────────────────────────────

class TestPipelineReturn:
    def test_returns_rag_result(self):
        rag = _rag_mock("COVID-19 causes pneumonia.")
        with patch("model.model.DBRetriever"), \
             patch("model.model.QA"), \
             patch("model.model.RetrievalQA") as MockRAG:
            MockRAG.from_chain_type.return_value = rag
            result = pipeline(api_key="key", question="What does COVID cause?", hf_auth="token")
        assert result == "COVID-19 causes pneumonia."

    def test_result_key_is_queried(self):
        rag = _rag_mock("answer")
        with patch("model.model.DBRetriever"), \
             patch("model.model.QA"), \
             patch("model.model.RetrievalQA") as MockRAG:
            MockRAG.from_chain_type.return_value = rag
            pipeline(api_key="k", question="Q?", hf_auth="t")
        rag.__getitem__.assert_called_with("result")


# ── DBRetriever construction ──────────────────────────────────────────────────

class TestPipelineRetrieverConstruction:
    def test_passes_api_key_to_retriever(self):
        with patch("model.model.DBRetriever") as MockDB, \
             patch("model.model.QA"), \
             patch("model.model.RetrievalQA") as MockRAG:
            MockRAG.from_chain_type.return_value = _rag_mock("ans")
            pipeline(api_key="my-pinecone-key", question="Q?", hf_auth="token")
        _, kwargs = MockDB.call_args
        assert kwargs["api_key"] == "my-pinecone-key"

    def test_passes_hf_auth_to_retriever(self):
        with patch("model.model.DBRetriever") as MockDB, \
             patch("model.model.QA"), \
             patch("model.model.RetrievalQA") as MockRAG:
            MockRAG.from_chain_type.return_value = _rag_mock("ans")
            pipeline(api_key="k", question="Q?", hf_auth="my-hf-token")
        _, kwargs = MockDB.call_args
        assert kwargs["hf_auth"] == "my-hf-token"

    def test_passes_index_name_to_retriever(self):
        with patch("model.model.DBRetriever") as MockDB, \
             patch("model.model.QA"), \
             patch("model.model.RetrievalQA") as MockRAG:
            MockRAG.from_chain_type.return_value = _rag_mock("ans")
            pipeline(api_key="k", question="Q?", hf_auth="t", index_name="my-index")
        _, kwargs = MockDB.call_args
        assert kwargs["index_name"] == "my-index"


# ── QA construction ───────────────────────────────────────────────────────────

class TestPipelineQAConstruction:
    def test_passes_question_as_prompt(self):
        with patch("model.model.DBRetriever"), \
             patch("model.model.QA") as MockQA, \
             patch("model.model.RetrievalQA") as MockRAG:
            MockRAG.from_chain_type.return_value = _rag_mock("ans")
            pipeline(api_key="k", question="What is aspirin?", hf_auth="t")
        _, kwargs = MockQA.call_args
        assert kwargs["prompt"] == "What is aspirin?"

    def test_passes_hf_auth_to_qa(self):
        with patch("model.model.DBRetriever"), \
             patch("model.model.QA") as MockQA, \
             patch("model.model.RetrievalQA") as MockRAG:
            MockRAG.from_chain_type.return_value = _rag_mock("ans")
            pipeline(api_key="k", question="Q?", hf_auth="hf-secret")
        _, kwargs = MockQA.call_args
        assert kwargs["hf_auth"] == "hf-secret"


# ── RetrievalQA construction ──────────────────────────────────────────────────

class TestPipelineRAGConstruction:
    def test_retrieval_qa_built_with_chain_type_stuff(self):
        with patch("model.model.DBRetriever"), \
             patch("model.model.QA"), \
             patch("model.model.RetrievalQA") as MockRAG:
            MockRAG.from_chain_type.return_value = _rag_mock("ans")
            pipeline(api_key="k", question="Q?", hf_auth="t")
        _, kwargs = MockRAG.from_chain_type.call_args
        assert kwargs["chain_type"] == "stuff"

    def test_k_controls_retriever_search_kwargs(self):
        mock_db_instance = MagicMock()
        with patch("model.model.DBRetriever", return_value=mock_db_instance), \
             patch("model.model.QA"), \
             patch("model.model.RetrievalQA") as MockRAG:
            MockRAG.from_chain_type.return_value = _rag_mock("ans")
            pipeline(api_key="k", question="Q?", hf_auth="t", k=5)
        mock_db_instance.vectorstore_db.as_retriever.assert_called_once_with(
            search_kwargs={"k": 5}
        )

    def test_default_k_is_10(self):
        mock_db_instance = MagicMock()
        with patch("model.model.DBRetriever", return_value=mock_db_instance), \
             patch("model.model.QA"), \
             patch("model.model.RetrievalQA") as MockRAG:
            MockRAG.from_chain_type.return_value = _rag_mock("ans")
            pipeline(api_key="k", question="Q?", hf_auth="t")
        mock_db_instance.vectorstore_db.as_retriever.assert_called_once_with(
            search_kwargs={"k": 10}
        )
