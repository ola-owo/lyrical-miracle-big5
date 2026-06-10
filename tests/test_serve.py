from collections.abc import Iterator
from typing import Any

import litserve as ls
import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from sentence_transformers import SentenceTransformer

from big5.globalvars import BIG5_TRAITS
from big5.serve import BigFiveAPI


class FakeEmbedding(list[float]):
    def tolist(self) -> list[float]:
        return list(self)


class FakeSentenceTransformer:
    def encode(
        self,
        texts: list[str],
        convert_to_numpy: bool = True,
        normalize_embeddings: bool = False,
    ) -> list[FakeEmbedding]:
        assert convert_to_numpy is True
        offset = 10.0 if normalize_embeddings else 0.0
        return [
            FakeEmbedding([offset + index + trait_index / 10 for trait_index in range(len(BIG5_TRAITS))])
            for index, _text in enumerate(texts)
        ]


@pytest.fixture()
def api() -> BigFiveAPI:
    api = BigFiveAPI(api_path="/predict")
    api.model = FakeSentenceTransformer() # pyright: ignore[reportAttributeAccessIssue]
    return api


@pytest.fixture()
def client(api: BigFiveAPI) -> Iterator[TestClient]:
    app = FastAPI()

    @app.get("/healthcheck")
    def healthcheck() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/predict")
    def predict(payload: dict[str, Any]) -> dict[str, Any]:
        try:
            request = api.decode_request(payload)
            output = api.predict(request)
            return api.encode_response(output)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    with TestClient(app) as test_client:
        yield test_client


def test_litserve_registers_vertex_routes() -> None:
    server = ls.LitServer(
        BigFiveAPI(api_path="/predict"),
        accelerator="cpu",
        devices=1,
        healthcheck_path="/healthcheck",
    )

    routes = {
        route.path: route.methods # pyright: ignore[reportAttributeAccessIssue]
        for route in server.app.routes
        if hasattr(route, "methods")
    }
    assert "POST" in routes["/predict"]
    assert "GET" in routes["/healthcheck"]


def test_healthcheck_endpoint(client: TestClient) -> None:
    response = client.get("/healthcheck")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_accepts_string_instances(client: TestClient) -> None:
    response = client.post(
        "/predict",
        json={"instances": ["first sample", "second sample"]},
    )

    assert response.status_code == 200
    assert response.json() == {
        "predictions": [
            {
                "Openness": 0.0,
                "Conscientiousness": 0.1,
                "Extraversion": 0.2,
                "Agreeableness": 0.3,
                "Neuroticism": 0.4,
            },
            {
                "Openness": 1.0,
                "Conscientiousness": 1.1,
                "Extraversion": 1.2,
                "Agreeableness": 1.3,
                "Neuroticism": 1.4,
            },
        ]
    }


def test_predict_accepts_text_object_instances_and_include_text(client: TestClient) -> None:
    response = client.post(
        "/predict",
        json={
            "instances": [{"text": "curious"}, {"text": "careful"}],
            "parameters": {"include_text": True},
        },
    )

    assert response.status_code == 200
    predictions = response.json()["predictions"]
    assert predictions[0]["text"] == "curious"
    assert predictions[1]["text"] == "careful"
    assert list(predictions[0]) == ["text", *BIG5_TRAITS]


def test_predict_honors_normalize_embeddings_parameter(client: TestClient) -> None:
    response = client.post(
        "/predict",
        json={
            "instances": ["normalized"],
            "parameters": {"normalize_embeddings": True},
        },
    )

    assert response.status_code == 200
    assert response.json()["predictions"] == [
        {
            "Openness": 10.0,
            "Conscientiousness": 10.1,
            "Extraversion": 10.2,
            "Agreeableness": 10.3,
            "Neuroticism": 10.4,
        }
    ]


@pytest.mark.parametrize(
    ("payload", "expected_detail"),
    [
        ({}, "Request body must include a non-empty 'instances' array."),
        ({"instances": []}, "Request body must include a non-empty 'instances' array."),
        ({"instances": [{"body": "missing text"}]}, "Each instance must be a string"),
        ({"instances": ["ok"], "parameters": []}, "'parameters' must be a JSON object"),
    ],
)
def test_predict_rejects_invalid_vertex_payloads(
    client: TestClient,
    payload: dict[str, object],
    expected_detail: str,
) -> None:
    response = client.post("/predict", json=payload)

    assert response.status_code == 400
    assert expected_detail in response.text
