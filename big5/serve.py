import os
from dataclasses import dataclass
from typing import Any

import litserve as ls
from sentence_transformers import SentenceTransformer

from big5.globalvars import BIG5_TRAITS, LORA_MODEL


@dataclass(frozen=True)
class VertexPredictionRequest:
    texts: list[str]
    include_text: bool
    normalize_embeddings: bool


def _to_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _coerce_text(instance: Any) -> str:
    if isinstance(instance, str):
        return instance
    if isinstance(instance, dict) and isinstance(instance.get("text"), str):
        return instance["text"]
    raise ValueError("Each instance must be a string or an object with a string 'text' field.")


class BigFiveAPI(ls.LitAPI):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def setup(self, device: str) -> None:
        self.model = SentenceTransformer(LORA_MODEL, device=str(device))

    def decode_request(self, request: dict[str, Any], **kwargs) -> VertexPredictionRequest:
        if not isinstance(request, dict):
            raise ValueError("Request body must be a JSON object.")

        instances = request.get("instances")
        if not isinstance(instances, list) or not instances:
            raise ValueError("Request body must include a non-empty 'instances' array.")

        parameters = request.get("parameters", {})
        if not isinstance(parameters, dict):
            raise ValueError("'parameters' must be a JSON object when provided.")

        return VertexPredictionRequest(
            texts=[_coerce_text(instance) for instance in instances],
            include_text=_to_bool(parameters.get("include_text"), default=False),
            normalize_embeddings=_to_bool(parameters.get("normalize_embeddings"), default=False),
        )

    def predict(self, request: VertexPredictionRequest, **kwargs) -> list[dict[str, Any]]: # pyright: ignore[reportIncompatibleMethodOverride]
        embeddings = self.model.encode(
            request.texts,
            convert_to_numpy=True,
            normalize_embeddings=request.normalize_embeddings,
        )

        predictions: list[dict[str, Any]] = []
        for text, embedding in zip(request.texts, embeddings, strict=True):
            scores = {
                trait: float(score)
                for trait, score in zip(BIG5_TRAITS, embedding.tolist(), strict=True)
            }
            if request.include_text:
                scores = {"text": text, **scores}
            predictions.append(scores)
        return predictions

    def encode_response(self, output: list[dict[str, Any]], **kwargs) -> dict[str, Any]:
        return {"predictions": output}


def main() -> None:
    port = int(os.getenv("AIP_HTTP_PORT", os.getenv("PORT", "8000")))
    predict_route = os.getenv("AIP_PREDICT_ROUTE", "/predict")
    health_route = os.getenv("AIP_HEALTH_ROUTE", "/health")
    max_payload_size = int(os.getenv("MAX_PAYLOAD_SIZE", "1500000"))

    server = ls.LitServer(
        BigFiveAPI(api_path=predict_route),
        accelerator=os.getenv("ACCELERATOR", "auto"), # pyright: ignore[reportArgumentType]
        healthcheck_path=health_route,
        max_payload_size=max_payload_size,
    )
    server.run(
        port=port,
        generate_client_file=False,
    )


if __name__ == "__main__":
    main()
