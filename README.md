# LitServe Vertex AI Serving

This repository includes a LitServe server for the configured `LORA_MODEL`.
It implements Vertex AI's `/predict` contract: requests use an `instances` array
and responses return a `predictions` array.

## Request parameters

The [`endpoints.predict`](https://docs.cloud.google.com/gemini-enterprise-agent-platform/reference/rest/v1/projects.locations.endpoints/predict)
endpoint specifies a required `instances` field which is a list of model inputs,
and an optional `parameters` field which is a dictionary of extra inference parameters.

`instances` can contain strings or `{"text": string}` objects.

Supported `parameters` keys are:

- `include_text`: boolean indicating whether to include the input text in the response
- `normalize_embeddings`: boolean indicating whether to normalize outputs to have a norm of 1

Example request:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "instances": [
      "I love meeting new people.",
      {"text": "I often worry about small things."}
    ],
    "parameters": {
      "include_text": true,
      "normalize_embeddings": false
    }
  }'
```

The response shape is:

```json
{
  "predictions": [
    {
      "text": "I love meeting new people.",
      "Openness": 0.0,
      "Conscientiousness": 0.0,
      "Extraversion": 0.0,
      "Agreeableness": 0.0,
      "Neuroticism": 0.0
    }
  ]
}
```

## Docker

Build and run the Docker container as usual:

```bash
docker buildx build -t lyrical-miracle-big5:latest .
docker run --rm -p 8080:8080 lyrical-miracle-big5:latest
```

**NOTE:** The default Pytorch includes Cuda 12.6 support.
If you wish to use the CPU-only version of PyTorch,
specify `--build-arg UV_INDEX="https://download.pytorch.org/whl/cpu"`.
Or use any of the other PyTorch package indexes listed on
[pytorch.org](https://pytorch.org/get-started/locally/)


For private Hugging Face models, set the environment variable `HF_TOKEN`
at runtime with your Hugging Face API key.

## Vertex AI Settings

Use these container fields when registering the model:

- `containerSpec.ports.containerPort`: `8080`
- `containerSpec.predictRoute`: `/predict`
- `containerSpec.healthRoute`: `/health`

The server also honors Vertex-provided environment variables:
`AIP_HTTP_PORT`, `AIP_PREDICT_ROUTE`, and `AIP_HEALTH_ROUTE`.
