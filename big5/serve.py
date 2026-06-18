import os
from typing import Any
import logging

import litserve as ls
from sentence_transformers import SentenceTransformer
from pydantic import (
    BaseModel,
    ConfigDict,
    RootModel,
    Field,
    TypeAdapter,
    model_validator,
)

from big5.globalvars import BIG5_TRAITS_SHORT, LORA_MODEL

bool_adapter = TypeAdapter(bool)

DEBUG_MODE = bool_adapter.validate_python(os.getenv('BIG5_DEBUG', False))
BATCH_MODE = bool_adapter.validate_python(os.getenv('BATCH_MODE', True))


class RequestText(RootModel[str]):
    @model_validator(mode='before')
    @classmethod
    def extract_str(cls, text):
        if isinstance(text, dict) and 'text' in text:
            return text['text']
        return text


class VertexPredictionRequest(BaseModel):
    model_config = ConfigDict(frozen=True, extra='forbid')
    texts: list[RequestText] = Field(min_length=1)
    include_text: bool = False
    include_traits: bool = True
    normalize_embeddings: bool = False


class BigFiveAPI(ls.LitAPI):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def setup(self, device: str) -> None:
        self.model = SentenceTransformer(LORA_MODEL, device=str(device))

    def decode_request(
        self, request: dict[str, Any], **kwargs
    ) -> VertexPredictionRequest:
        log = logging.getLogger(__name__)
        if not isinstance(request, dict):
            if DEBUG_MODE:
                log.error(f'Invalid request format: {type(request)}')
            raise ValueError('Request body must be a JSON object.')

        instances = request.get('instances', [])
        log.info(f'Received instance: {instances}')
        if not isinstance(instances, list) or not instances:
            raise ValueError("Request body must include a non-empty 'instances' array.")

        parameters = request.get('parameters', {})
        if not isinstance(parameters, dict):
            raise ValueError("'parameters' must be a JSON object when provided.")

        return VertexPredictionRequest(texts=instances, **parameters)

    def _predict_batch(
        self, request: VertexPredictionRequest | list[VertexPredictionRequest], **kwargs
    ) -> list[dict[str, Any]]:  # pyright: ignore[reportIncompatibleMethodOverride]
        embeddings = self.model.encode(
            request.texts,
            convert_to_numpy=True,
            normalize_embeddings=request.normalize_embeddings,
        )

        predictions: list[dict[str, Any]] = []
        for text, embedding in zip(request.texts, embeddings, strict=True):
            scores = [float(score) for score in embedding.tolist()]
            if request.include_traits:
                scores = {
                    trait: float(score)
                    for trait, score in zip(BIG5_TRAITS_SHORT, scores, strict=True)
                }
            if request.include_text:
                scores = {'text': text, 'prediction': scores}
            predictions.append(scores)
        return predictions

    def predict(
        self, request: VertexPredictionRequest, **kwargs
    ) -> list[dict[str, Any]]:  # pyright: ignore[reportIncompatibleMethodOverride]
        if isinstance(request, list):
            return [self._predict_batch(batch, **kwargs) for batch in request]
        else:
            return self._predict_batch(request, **kwargs)

    def encode_response(self, output: list[dict[str, Any]], **kwargs) -> dict[str, Any]:
        return {'predictions': output}


def main() -> None:
    log = logging.getLogger(__name__)
    port = int(os.getenv('AIP_HTTP_PORT', os.getenv('PORT', '8000')))
    predict_route = os.getenv('AIP_PREDICT_ROUTE', '/predict')
    health_route = os.getenv('AIP_HEALTH_ROUTE', '/health')
    max_payload_size = int(os.getenv('MAX_PAYLOAD_SIZE', '1500000'))
    max_batch_size = int(os.getenv('BATCH_SIZE', '32')) if BATCH_MODE else 1
    if DEBUG_MODE:
        if BATCH_MODE:
            log.info(f'Running with batch mode ENABLED (N={max_batch_size})')
        else:
            log.info('Running with batch mode DISABLED')

    server = ls.LitServer(
        BigFiveAPI(api_path=predict_route, max_batch_size=max_batch_size),
        accelerator=os.getenv('ACCELERATOR', 'auto'),
        healthcheck_path=health_route,
        max_payload_size=max_payload_size,
        timeout=int(os.getenv('INFERENCE_TIMEOUT', '30')),
    )
    server.run(
        port=port,
        generate_client_file=False,
    )


if __name__ == '__main__':
    main()
