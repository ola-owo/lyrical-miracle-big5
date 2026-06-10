from collections.abc import Iterator
from typing import Any

import litserve as ls
import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from big5.globalvars import BIG5_TRAITS
from big5.serve import BigFiveAPI


class FakeEmbedding(list[float]):
    def tolist(self) -> list[float]:
        return list(self)

    def normalize(self):
        scalar = 1.0 / sum(self)
        return FakeEmbedding([scalar * x for x in self])


class FakeSentenceTransformer:
    def encode(
        self,
        texts: list[str],
        convert_to_numpy: bool = True,
        normalize_embeddings: bool = False,
    ) -> list[FakeEmbedding]:
        assert convert_to_numpy
        n_traits = len(BIG5_TRAITS)
        outputs = [
            FakeEmbedding([10 * i + trait_index for trait_index in range(n_traits)])
            for i in range(len(texts))
        ]
        if normalize_embeddings:
            outputs = [out.normalize() for out in outputs]
        return outputs


@pytest.fixture()
def api() -> BigFiveAPI:
    api = BigFiveAPI(api_path='/predict')
    api.model = FakeSentenceTransformer()  # pyright: ignore[reportAttributeAccessIssue]
    return api


@pytest.fixture()
def client(api: BigFiveAPI) -> Iterator[TestClient]:
    app = FastAPI()

    @app.post('/predict')
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
        BigFiveAPI(api_path='/predict'),
        accelerator='cpu',
        devices=1,
        healthcheck_path='/health',
    )

    routes = {
        route.path: route.methods  # pyright: ignore[reportAttributeAccessIssue]
        for route in server.app.routes
        if hasattr(route, 'methods')
    }
    assert 'POST' in routes['/predict']
    assert 'GET' in routes['/health']


def test_predict_accepts_string_instances(client: TestClient) -> None:
    response = client.post(
        '/predict',
        json={'instances': ['first sample', 'second sample']},
    )

    assert response.status_code == 200
    assert response.json() == {
        'predictions': [
            {
                'Openness': 0,
                'Conscientiousness': 1,
                'Extraversion': 2,
                'Agreeableness': 3,
                'Neuroticism': 4,
            },
            {
                'Openness': 10,
                'Conscientiousness': 11,
                'Extraversion': 12,
                'Agreeableness': 13,
                'Neuroticism': 14,
            },
        ]
    }


def test_predict_accepts_text_object_instances_and_include_text(
    client: TestClient,
) -> None:
    input_texts = ['curious', 'careful']
    response = client.post(
        '/predict',
        json={
            'instances': [{'text': text} for text in input_texts],
            'parameters': {'include_text': True},
        },
    )

    assert response.status_code == 200
    predictions = response.json()['predictions']
    for text, pred in zip(input_texts, predictions):
        assert pred['text'] == text
        assert set(pred.keys()) == set(BIG5_TRAITS + ['text'])


def test_predict_honors_normalize_embeddings_parameter(client: TestClient) -> None:
    response = client.post(
        '/predict',
        json={
            'instances': ['normalized'],
            'parameters': {'normalize_embeddings': True},
        },
    )

    vec_sum = sum(range(5))
    assert response.status_code == 200
    assert response.json()['predictions'] == [
        {
            'Openness': pytest.approx(0 / vec_sum),
            'Conscientiousness': pytest.approx(1 / vec_sum),
            'Extraversion': pytest.approx(2 / vec_sum),
            'Agreeableness': pytest.approx(3 / vec_sum),
            'Neuroticism': pytest.approx(4 / vec_sum),
        }
    ]


@pytest.mark.parametrize(
    ('payload', 'expected_detail'),
    [
        ({}, "Request body must include a non-empty 'instances' array."),
        ({'instances': []}, "Request body must include a non-empty 'instances' array."),
        ({'instances': [{'body': 'missing text'}]}, 'Each instance must be a string'),
        ({'instances': ['ok'], 'parameters': []}, "'parameters' must be a JSON object"),
    ],
)
def test_predict_rejects_invalid_vertex_payloads(
    client: TestClient,
    payload: dict[str, object],
    expected_detail: str,
) -> None:
    response = client.post('/predict', json=payload)

    assert response.status_code == 400
    assert expected_detail in response.text
