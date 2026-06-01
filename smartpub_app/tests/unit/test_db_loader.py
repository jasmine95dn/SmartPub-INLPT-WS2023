import pytest
from unittest.mock import MagicMock, patch
from model.db_loader import readout_triplets, PineconeVDB


# ── readout_triplets ──────────────────────────────────────────────────────────

class TestReadoutTriplets:
    def test_empty_list_returns_empty_string(self):
        assert readout_triplets([]) == ""

    def test_single_triplet_formatted_correctly(self):
        triplets = [{"head": "COVID-19", "type": "causes", "tail": "pneumonia"}]
        result = readout_triplets(triplets)
        assert "|COVID-19 - causes - pneumonia|" in result

    def test_multiple_triplets_all_present(self):
        triplets = [
            {"head": "A", "type": "rel1", "tail": "B"},
            {"head": "C", "type": "rel2", "tail": "D"},
        ]
        result = readout_triplets(triplets)
        assert "|A - rel1 - B|" in result
        assert "|C - rel2 - D|" in result

    def test_all_fields_present_in_output(self, sample_triplets):
        result = readout_triplets(sample_triplets)
        for t in sample_triplets:
            assert t["head"] in result
            assert t["type"] in result
            assert t["tail"] in result


# ── PineconeVDB fixture ───────────────────────────────────────────────────────

@pytest.fixture
def vdb(monkeypatch):
    """PineconeVDB with env vars set and external calls mocked via stubs."""
    monkeypatch.setenv("PINECONE_API_KEY", "test-pinecone-key")
    monkeypatch.setenv("PINECONE_ENVIRONMENT", "us-east-1-aws")
    monkeypatch.setenv("HF_AUTH", "hf-test-token")
    instance = PineconeVDB(embedding_model_name="sentence-transformers/all-MiniLM-L6-v2")
    # pc_db is the shared return_value of the Pinecone mock class; reset call
    # history so previous tests don't bleed into the next.
    instance.pc_db.reset_mock()
    return instance


# ── PineconeVDB.create_pinecone_index ─────────────────────────────────────────

class TestCreatePineconeIndex:
    def test_creates_index_when_not_exists(self, vdb):
        vdb.pc_db.list_indexes.return_value.names.return_value = []
        vdb.embedding_model.embed_documents.return_value = [[0.1] * 384]

        vdb.create_pinecone_index("new-index")

        vdb.pc_db.create_index.assert_called_once()
        args, kwargs = vdb.pc_db.create_index.call_args
        assert args[0] == "new-index"
        assert kwargs.get("dimension") == 384

    def test_skips_creation_when_index_already_exists(self, vdb):
        vdb.pc_db.list_indexes.return_value.names.return_value = ["existing-index"]

        vdb.create_pinecone_index("existing-index")

        vdb.pc_db.create_index.assert_not_called()

    def test_uses_cosine_metric_by_default(self, vdb):
        vdb.pc_db.list_indexes.return_value.names.return_value = []
        vdb.embedding_model.embed_documents.return_value = [[0.0] * 768]

        vdb.create_pinecone_index("my-index")

        _, kwargs = vdb.pc_db.create_index.call_args
        assert kwargs.get("metric") == "cosine"


# ── PineconeVDB.find_most_similar_docs ────────────────────────────────────────

class TestFindMostSimilarDocs:
    def test_returns_similarity_search_result(self, vdb):
        mock_docs = [MagicMock(page_content="doc1"), MagicMock(page_content="doc2")]
        mock_index = MagicMock()

        import model.db_loader as db_loader
        original_vs = db_loader.vectorstore_pc
        mock_vs_cls = MagicMock()
        mock_vs_cls.return_value.similarity_search.return_value = mock_docs
        db_loader.vectorstore_pc = mock_vs_cls
        try:
            result = vdb.find_most_similar_docs("COVID treatment", mock_index, num_of_chunks=2)
            assert result == mock_docs
        finally:
            db_loader.vectorstore_pc = original_vs

    def test_passes_k_to_similarity_search(self, vdb):
        mock_index = MagicMock()

        import model.db_loader as db_loader
        original_vs = db_loader.vectorstore_pc
        mock_vs_cls = MagicMock()
        db_loader.vectorstore_pc = mock_vs_cls
        try:
            vdb.find_most_similar_docs("query", mock_index, num_of_chunks=7)
            mock_vs_cls.return_value.similarity_search.assert_called_once_with("query", k=7)
        finally:
            db_loader.vectorstore_pc = original_vs


# ── PineconeVDB.push_data_to_index ────────────────────────────────────────────

class TestPushDataToIndex:
    def test_calls_upsert(self, vdb, sample_entry):
        vdb.embedding_model.embed_documents.return_value = [[0.1] * 384]

        vdb.push_data_to_index("smartpub", [sample_entry], namespace="test-ns")

        vdb.pc_db.Index.return_value.upsert.assert_called_once()

    def test_opens_correct_index(self, vdb, sample_entry):
        vdb.embedding_model.embed_documents.return_value = [[0.1] * 384]

        vdb.push_data_to_index("smartpub", [sample_entry], namespace="test-ns")

        vdb.pc_db.Index.assert_called_with("smartpub")

    def test_upsert_called_with_namespace(self, vdb, sample_entry):
        vdb.embedding_model.embed_documents.return_value = [[0.1] * 384]

        vdb.push_data_to_index("smartpub", [sample_entry], namespace="my-namespace")

        _, kwargs = vdb.pc_db.Index.return_value.upsert.call_args
        assert kwargs.get("namespace") == "my-namespace"
