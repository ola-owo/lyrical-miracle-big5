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

- `normalize_embeddings`: boolean indicating whether to normalize outputs to have a norm of 1 (default `false`)
- `include_text`: boolean indicating whether to include the input text in the response (default `false`)
- `include_traits`: output a JSON object with trait labels if `true`, or an array without labels if `false` (default `true`)

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
      "prediction": {
        "OPN": 0.0,
        "CON": 0.0,
        "EXT": 0.0,
        "AGR": 0.0,
        "NEU": 0.0
      }
    }
  ]
}
```

## Docker

### Build the image

Build and run the Docker container as usual:

```bash
docker buildx build -t lyrical-miracle-big5:cpu .
docker run --rm -p 8080:8080 lyrical-miracle-big5:cpu
```

**NOTE:** The Dockerfile uses CPU-only Pytorch by default (`pylock.cpu.toml`).
To build with CUDA support,`--build-arg PYLOCK=pylock.cu13.toml`
and use the `cu13` tag.

**NOTE:** You may wish to set the environment variable `HF_TOKEN`
as Hugging Face API key which will enable faster model downloading
and increased rate limits.

## Google AI Platform

Follow these steps to deploy this model on Google Cloud.

### Push to Artifact Registry

Tag the image

```sh
docker tag lyrical-miracle-big5:cpu $AR_TAG_NAME
```

where `$AR_TAG_NAME` is the full path of your artifact repository
(e.g. `us-central1.docker.pkg.dev/my-project/big5-repo`)

Then push the tagged image to GCP:

```sh
docker push $AR_TAG_NAME
```

### Deploy the model

Import the container image into Model Registry
Set these fields:

- `containerSpec.ports.containerPort`: `8080`
- `containerSpec.predictRoute`: `/predict`
- `containerSpec.healthRoute`: `/health`

Or set the environment variables:
`AIP_HTTP_PORT`, `AIP_PREDICT_ROUTE`, and `AIP_HEALTH_ROUTE`
to use non-default values.
