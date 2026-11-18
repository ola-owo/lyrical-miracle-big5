# Big-5 AI Inference Server

This repository includes a LitServe server for the configured Sentence Transformers model.
It implements Vertex AI's `/predict` contract:
requests use an `instances` array and responses return a `predictions` array.

## Request parameters

The
[`endpoints.predict`](https://docs.cloud.google.com/gemini-enterprise-agent-platform/reference/rest/v1/projects.locations.endpoints/predict)
endpoint specifies a required `instances` field which is a list of model inputs, and an optional
`parameters` field which is a dictionary of extra inference parameters.

`instances` can contain strings or `{"text": string}` objects.

Supported `parameters` keys are:

- `normalize_embeddings`: boolean indicating whether to normalize outputs to have a norm of 1
  (default `false`)
- `include_text`: boolean indicating whether to include the input text in the response
  (default `false`)
- `include_traits`: output a JSON object with trait labels if `true`,
  or an array without labels if `false` (default `true`)

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

## Model storage

Serving never downloads from Hugging Face. The model must be available as a
standalone directory:

- `BIG5_MODEL_URI`: a `gs://` prefix or local directory.
- `AIP_STORAGE_URI`: set automatically by Vertex AI when the model is deployed
  with an `artifactUri`.

`BIG5_MODEL_URI` takes precedence. If neither is set, the server exits with an
error. `gs://` downloads use Application Default Credentials (ADC) and are
cached in `BIG5_MODEL_CACHE` (default: a directory under the system temp dir).
Use a new versioned prefix for each model release; the cache is keyed by URI and
does not notice overwritten objects.

Model artifacts are downloaded and loaded before LitServe binds the HTTP port.
If startup takes longer than the Vertex AI liveness probe window, configure a
startup probe on the model's container spec.

### Export a standalone model

The Hugging Face repository contains a PEFT adapter over the gated
`google/embeddinggemma-300m` base model. Merge it before uploading:

```bash
uv run --extra cpu export-model \
  --source ola-owo/big5-sentence-transformer-lora \
  --output models/export/big5-sentence-transformer-lora/v1
```

The export command requires Hugging Face access once to download the gated base
model. It verifies the result with `HF_HUB_OFFLINE=1` before finishing.

### Upload

```bash
gcloud storage cp -r models/export/big5-sentence-transformer-lora/v1/* \
  gs://<bucket>/big5/models/big5-sentence-transformer-lora/v1/
```

Keep the bucket private. Confirm the layout:

```bash
gcloud storage ls gs://<bucket>/big5/models/big5-sentence-transformer-lora/v1/modules.json
```

### Authentication

Local development:

```bash
gcloud auth application-default login
```

On Vertex AI, ADC is provided by the metadata server. For direct bucket access,
grant the runtime service account `roles/storage.objectViewer` on the bucket:

```bash
gcloud storage buckets add-iam-policy-binding gs://<bucket> \
  --member=serviceAccount:<runtime-sa>@<project>.iam.gserviceaccount.com \
  --role=roles/storage.objectViewer
```

If you deploy with an `artifactUri`, Vertex AI copies the artifacts to a
managed bucket, sets `AIP_STORAGE_URI`, and grants the service account read
access automatically. The Vertex AI service agent
(`service-<project-number>@gcp-sa-aiplatform.iam.gserviceaccount.com`) needs
`roles/storage.objectViewer` on the source bucket to perform that copy.

For direct `BIG5_MODEL_URI` access, note that the default prediction service
account is managed by Vertex AI and cannot be granted bucket permissions.
Configure a custom service account for the deployed model and grant it
`roles/storage.objectViewer` on the bucket, or use `artifactUri` instead.

### Vertex AI artifact option

Set `artifactUri` to the GCS prefix when importing the model. The container
reads the URI from `AIP_STORAGE_URI`; no bucket name or `BIG5_MODEL_URI` is
needed in the image.

## Docker

### Build the image

Build and run the Docker container as usual:

```bash
# build with CPU pytorch
docker buildx build -t lyrical-miracle-big5:cpu .
# or build with CUDA pytorch
docker buildx build --build-arg TORCH_BACKEND=cuda -t lyrical-miracle-big5:cuda .

docker run --rm -p 8080:8080 lyrical-miracle-big5
```

**NOTE:** The container needs Application Default Credentials when the model is
loaded from `gs://`. For local Docker testing, mount your ADC directory:

```bash
docker run --rm -p 8080:8080 \
  -e BIG5_MODEL_URI=gs://<bucket>/big5/models/big5-sentence-transformer-lora/v1 \
  -v ~/.config/gcloud:/root/.config/gcloud:ro \
  lyrical-miracle-big5:cpu
```

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

Import the container image into Model Registry Set these fields:

- `containerSpec.ports.containerPort`: `8080`
- `containerSpec.predictRoute`: `/predict`
- `containerSpec.healthRoute`: `/health`

Then choose one model source:

- Set `containerSpec.env.BIG5_MODEL_URI` to the GCS model prefix and grant the
  runtime service account `roles/storage.objectViewer` on the bucket.
- Or set `artifactUri` to the GCS model prefix. Vertex AI copies the artifacts
  and sets `AIP_STORAGE_URI` automatically.

Or set the environment variables: `AIP_HTTP_PORT`, `AIP_PREDICT_ROUTE`,
and `AIP_HEALTH_ROUTE` to use non-default values.
