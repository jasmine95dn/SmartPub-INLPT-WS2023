"""
System tests — exercise the full HTTP interface of the Flask app.
External ML inference (pipeline) is mocked so these tests run without
Pinecone or HuggingFace credentials and without loading the LLM.

Run with:   poetry run pytest -m system -v
Excluded from normal CI via: pytest -m "not system"
"""
import pytest
from unittest.mock import patch


@pytest.fixture(scope="module")
def system_app():
    from app import app
    app.config["TESTING"] = True
    return app


@pytest.fixture
def client(system_app):
    with system_app.test_client() as c:
        yield c


# ── Chat page ─────────────────────────────────────────────────────────────────

@pytest.mark.system
class TestChatPageLoads:
    def test_index_returns_200(self, client):
        response = client.get("/")
        assert response.status_code == 200

    def test_index_returns_html_document(self, client):
        response = client.get("/")
        assert response.content_type.startswith("text/html")

    def test_index_contains_form_input(self, client):
        response = client.get("/")
        assert b'name="msg"' in response.data

    def test_index_contains_send_button(self, client):
        response = client.get("/")
        assert b"send" in response.data.lower()


# ── Chat endpoint — happy path ────────────────────────────────────────────────

@pytest.mark.system
class TestChatEndpointHappyPath:
    def test_post_with_question_returns_200(self, client):
        with patch("app.pipeline", return_value="COVID-19 is a respiratory illness."):
            response = client.post("/get", data={"msg": "What is COVID-19?"})
        assert response.status_code == 200

    def test_response_body_is_pipeline_output(self, client):
        answer = "Aspirin treats headaches by inhibiting COX enzymes."
        with patch("app.pipeline", return_value=answer):
            response = client.post("/get", data={"msg": "How does aspirin work?"})
        assert response.get_data(as_text=True) == answer

    def test_pipeline_receives_question_text(self, client):
        with patch("app.pipeline", return_value="answer") as mock_pl:
            client.post("/get", data={"msg": "What is pneumonia?"})
        _, kwargs = mock_pl.call_args
        assert kwargs["question"] == "What is pneumonia?"

    def test_response_content_type_is_text(self, client):
        with patch("app.pipeline", return_value="plain text answer"):
            response = client.post("/get", data={"msg": "question"})
        assert "text" in response.content_type


# ── Chat endpoint — error handling ────────────────────────────────────────────

@pytest.mark.system
class TestChatEndpointErrors:
    def test_missing_msg_field_returns_500(self, client):
        response = client.post("/get", data={})
        assert response.status_code == 500

    def test_pipeline_runtime_error_returns_500(self, client):
        with patch("app.pipeline", side_effect=RuntimeError("model unavailable")):
            response = client.post("/get", data={"msg": "question"})
        assert response.status_code == 500

    def test_error_response_contains_error_message(self, client):
        with patch("app.pipeline", side_effect=ValueError("bad input")):
            response = client.post("/get", data={"msg": "question"})
        assert b"An error occurred" in response.data

    def test_pipeline_connection_error_returns_500(self, client):
        with patch("app.pipeline", side_effect=ConnectionError("Pinecone unreachable")):
            response = client.post("/get", data={"msg": "question"})
        assert response.status_code == 500


# ── Chat endpoint — multiple requests ────────────────────────────────────────

@pytest.mark.system
class TestChatEndpointSequential:
    def test_handles_multiple_questions_independently(self, client):
        answers = ["Answer one.", "Answer two.", "Answer three."]
        for i, expected in enumerate(answers):
            with patch("app.pipeline", return_value=expected):
                response = client.post("/get", data={"msg": f"Question {i + 1}?"})
            assert response.get_data(as_text=True) == expected

    def test_empty_string_question_handled(self, client):
        with patch("app.pipeline", return_value="I need more context."):
            response = client.post("/get", data={"msg": ""})
        assert response.status_code == 200
