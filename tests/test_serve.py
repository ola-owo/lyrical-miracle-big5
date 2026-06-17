from collections.abc import Iterator
from typing import Any

import litserve as ls
import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from big5.globalvars import BIG5_TRAITS_SHORT
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
        n_traits = len(BIG5_TRAITS_SHORT)
        outputs = [
            FakeEmbedding([10 * i + trait_index for trait_index in range(n_traits)])
            for i in range(len(texts))
        ]
        if normalize_embeddings:
            outputs = [out.normalize() for out in outputs]
        return outputs


@pytest.fixture(params=[1, 2], ids=['no_batch', 'batch'])
def api(request) -> BigFiveAPI:
    api = BigFiveAPI(api_path='/predict', max_batch_size=request.param)
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
                'OPN': 0,
                'CON': 1,
                'EXT': 2,
                'AGR': 3,
                'NEU': 4,
            },
            {
                'OPN': 10,
                'CON': 11,
                'EXT': 12,
                'AGR': 13,
                'NEU': 14,
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
    response_json = response.json()
    predictions = response_json['predictions']
    for text, pred in zip(input_texts, predictions):
        assert set(pred.keys()) == {'text', 'prediction'}
        assert pred['text'] == text
        assert set(pred['prediction'].keys()) == set(BIG5_TRAITS_SHORT)


def test_predict_normalize_embeddings(client: TestClient) -> None:
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
            'OPN': pytest.approx(0 / vec_sum),
            'CON': pytest.approx(1 / vec_sum),
            'EXT': pytest.approx(2 / vec_sum),
            'AGR': pytest.approx(3 / vec_sum),
            'NEU': pytest.approx(4 / vec_sum),
        }
    ]


def test_predict_no_include_traits_no_include_text(
    client: TestClient,
) -> None:
    input_texts = ['bold and brash']
    response = client.post(
        '/predict',
        json={
            'instances': [{'text': text} for text in input_texts],
            'parameters': {'include_traits': False},
        },
    )

    assert response.status_code == 200
    assert response.json()['predictions'] == [[0, 1, 2, 3, 4]]


def test_predict_no_include_traits_include_text(
    client: TestClient,
) -> None:
    input_texts = ['bold and brash']
    response = client.post(
        '/predict',
        json={
            'instances': [{'text': text} for text in input_texts],
            'parameters': {'include_text': True, 'include_traits': False},
        },
    )

    assert response.status_code == 200
    assert response.json()['predictions'] == [
        {
            'text': input_texts[0],
            'prediction': [0, 1, 2, 3, 4],
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
