import pytest
from unittest.mock import MagicMock
from model.qa_inference import QA


@pytest.fixture
def qa():
    return QA(prompt="What is COVID-19?", hf_auth="hf-test-token")


# ── QA.__init__ ───────────────────────────────────────────────────────────────


class TestQAInit:
    def test_stores_prompt(self, qa):
        assert qa.prompt == "What is COVID-19?"

    def test_default_task_is_text_generation(self, qa):
        assert qa.task == "text-generation"

    def test_default_model_is_llama(self, qa):
        assert "Llama" in qa.model_name or "llama" in qa.model_name

    def test_stores_hf_auth(self, qa):
        assert qa.hf_auth == "hf-test-token"

    def test_custom_task(self):
        q = QA(prompt="Q?", task="text2text-generation", hf_auth="token")
        assert q.task == "text2text-generation"

    def test_custom_model_name(self):
        q = QA(prompt="Q?", model_name="t5-small", hf_auth="token")
        assert q.model_name == "t5-small"


# ── QA.qa_inference — text-generation ────────────────────────────────────────


class TestQAInferenceTextGeneration:
    def test_sets_llm_attribute(self, qa):
        import model.qa_inference as qi_mod

        mock_pipeline_fn = MagicMock(return_value=MagicMock())
        mock_hf_pipeline_cls = MagicMock(return_value=MagicMock())

        original_pipeline = qi_mod.pipeline
        original_hf = qi_mod.HuggingFacePipeline
        qi_mod.pipeline = mock_pipeline_fn
        qi_mod.HuggingFacePipeline = mock_hf_pipeline_cls
        try:
            qa.qa_inference(
                task="text-generation", model_name="meta-llama/Llama-2-13b-chat-hf"
            )
            assert hasattr(qa, "llm")
        finally:
            qi_mod.pipeline = original_pipeline
            qi_mod.HuggingFacePipeline = original_hf

    def test_sets_qa_pipeline_attribute(self, qa):
        import model.qa_inference as qi_mod

        mock_result = MagicMock()
        mock_pipeline_fn = MagicMock(return_value=mock_result)

        original = qi_mod.pipeline
        qi_mod.pipeline = mock_pipeline_fn
        try:
            qa.qa_inference(
                task="text-generation", model_name="meta-llama/Llama-2-13b-chat-hf"
            )
            assert qa.qa_pipeline == mock_result
        finally:
            qi_mod.pipeline = original

    def test_calls_model_eval(self, qa):
        import model.qa_inference as qi_mod

        mock_model = MagicMock()
        qi_mod.AutoModelForCausalLM.from_pretrained.return_value = mock_model

        original = qi_mod.pipeline
        qi_mod.pipeline = MagicMock(return_value=MagicMock())
        try:
            qa.qa_inference(
                task="text-generation", model_name="meta-llama/Llama-2-13b-chat-hf"
            )
            mock_model.eval.assert_called_once()
        finally:
            qi_mod.pipeline = original


# ── QA.qa_inference — text2text-generation ───────────────────────────────────


class TestQAInferenceText2Text:
    def test_returns_generated_text(self):
        q = QA(prompt="Summarise COVID.", task="text2text-generation", hf_auth="token")

        import model.qa_inference as qi_mod

        mock_pipeline_fn = MagicMock(
            return_value=MagicMock(
                return_value=[{"generated_text": "COVID is a disease."}]
            )
        )
        original = qi_mod.pipeline
        qi_mod.pipeline = mock_pipeline_fn
        try:
            result = q.qa_inference(task="text2text-generation", model_name="t5-small")
            assert result == "COVID is a disease."
        finally:
            qi_mod.pipeline = original

    def test_calls_pipeline_with_correct_task(self):
        q = QA(prompt="Q?", task="text2text-generation", hf_auth="token")

        import model.qa_inference as qi_mod

        mock_pipeline_fn = MagicMock(
            return_value=MagicMock(return_value=[{"generated_text": "answer"}])
        )
        original = qi_mod.pipeline
        qi_mod.pipeline = mock_pipeline_fn
        try:
            q.qa_inference(task="text2text-generation", model_name="t5-small")
            args, kwargs = mock_pipeline_fn.call_args
            assert args[0] == "text2text-generation"
        finally:
            qi_mod.pipeline = original
