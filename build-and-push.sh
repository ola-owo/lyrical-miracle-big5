#!/usr/bin/env bash

: "${GCP_REGION:=us-central1}"
: "${GCP_PROJECT:=crucial-bucksaw-475302-s4}"
: "${GCP_REPO:=big5-server}"

: "${REPO:=${GCP_REGION}-docker.pkg.dev/${GCP_PROJECT}/${GCP_REPO}}"
: "${IMG:=lyrical-miracle-big5}"
: "${TAG:=cpu}"

echo "pushing ${IMG}:${TAG} -> ${REPO} ..."

docker buildx build --build-arg TORCH_BACKEND=${TAG} -t ${IMG}:${TAG} . && \
docker tag ${IMG}:${TAG} ${REPO}/${IMG}:${TAG} && \
docker push ${REPO}/${IMG}:${TAG}
