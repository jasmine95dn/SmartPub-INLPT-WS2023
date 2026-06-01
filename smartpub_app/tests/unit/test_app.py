import pytest
from unittest.mock import patch, MagicMock


@pytest.fixture(scope="module")
def flask_app():
    from app import app
    app.config["TESTING"] = True
    return app


@pytest.fixture
def client(flask_app):
    with flask_app.test_client() as c:
        yield c


# ── GET / ─────────────────────────────────────────────────────────────────────

class TestIndexRoute:
    def test_returns_200(self, client):
        response = client.get("/")
        assert response.status_code == 200

    def test_returns_html_content(self, client):
        response = client.get("/")
        assert b"html" in response.data.lower()

    def test_contains_smartpub_title(self, client):
        response = client.get("/")
        assert b"Smartpub" in response.data or b"smartpub" in response.data.lower()


# ── POST /get ────────────────────────────────────────────────────────────────

class TestChatRoute:
    def test_returns_pipeline_answer(self, client):
        with patch("app.pipeline", return_value="COVID-19 causes pneumonia."):
            response = client.post("/get", data={"msg": "What does COVID-19 cause?"})
        assert response.status_code == 200
        assert b"COVID-19" in response.data

    def test_response_matches_pipeline_output(self, client):
        with patch("app.pipeline", return_value="42 is the answer."):
            response = client.post("/get", data={"msg": "What is the answer?"})
        assert response.data == b"42 is the answer."

    def test_missing_msg_returns_500(self, client):
        response = client.post("/get", data={})
        assert response.status_code == 500

    def test_pipeline_exception_returns_500(self, client):
        with patch("app.pipeline", side_effect=RuntimeError("model load failed")):
            response = client.post("/get", data={"msg": "What is COVID?"})
        assert response.status_code == 500

    def test_error_response_contains_message(self, client):
        with patch("app.pipeline", side_effect=ValueError("bad input")):
            response = client.post("/get", data={"msg": "test"})
        assert b"An error occurred" in response.data

    def test_accepts_get_request(self, client):
        with patch("app.pipeline", return_value="answer"):
            response = client.get("/get", query_string={"msg": "question"})
        # GET is allowed by the route; missing form data triggers 500
        assert response.status_code in (200, 500)
