import logging
import os
from typing import Any, override

import litserve as ls
from fastapi import HTTPException
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    RootModel,
    TypeAdapter,
    model_validator,
)

from big5.globalvars import BIG5_TRAITS_SHORT, MODEL

bool_adapter = TypeAdapter(bool)

DEBUG_MODE = bool_adapter.validate_python(os.getenv('BIG5_DEBUG', 'False'))
BATCH_MODE = bool_adapter.validate_python(os.getenv('BATCH_MODE', 'True'))


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

    @override
    def setup(self, device: str) -> None:
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(MODEL, device=str(device))

    @override
    def decode_request(
        self, request: dict[str, Any], **kwargs
    ) -> VertexPredictionRequest:
        log = logging.getLogger(__name__)
        if not isinstance(request, dict):
            if DEBUG_MODE:
                log.error(f'Invalid request format: {type(request)}')
            raise HTTPException(400, 'Request body must be a JSON object.')

        instances = request.get('instances', [])
        log.info(f'Received instance: {instances}')
        if not isinstance(instances, list) or not instances:
            raise HTTPException(
                400, "Request body must include a non-empty 'instances' array."
            )

        parameters = request.get('parameters', {})
        if not isinstance(parameters, dict):
            raise HTTPException(
                400, "'parameters' must be a JSON object when provided."
            )

        return VertexPredictionRequest(texts=instances, **parameters)

    def _predict(
        self, request: VertexPredictionRequest
    ) -> list[dict[str, Any]] | list[list[float]]:
        texts = [t.root for t in request.texts]
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=request.normalize_embeddings,
        )

        predictions = []
        for text, embedding in zip(texts, embeddings, strict=True):
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

    @override
    def predict(
        self,
        request: VertexPredictionRequest | list[VertexPredictionRequest],
    ) -> dict[str, Any] | list[dict[str, Any]]:
        if isinstance(request, list):
            return [self._predict(batch) for batch in request]
        else:
            return self._predict(request)

    @override
    def encode_response(self, output: list[dict[str, Any]], **kwargs) -> dict[str, Any]:
        return {'predictions': output}


def main() -> None:
    import sentence_transformers  # noqa: F401  # fail fast if the model extra is missing

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
        accelerator=os.getenv('ACCELERATOR', 'auto'),  # pyright: ignore[reportArgumentType]
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
