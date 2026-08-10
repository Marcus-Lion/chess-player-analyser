#!/usr/bin/env bash
#
# Deploy the app to Google Cloud Run with the self-play flags enabled.
#
# Usage:
#   ./deploy-cloudrun.sh
#
# Optional environment variables:
#   SERVICE_NAME  Cloud Run service name (default: chess-player-analyser)
#   REGION        Cloud Run region (default: us-central1)
#
set -euo pipefail

SERVICE_NAME="${SERVICE_NAME:-chess-player-analyser}"
REGION="${REGION:-us-central1}"

gcloud run deploy "$SERVICE_NAME" \
  --source . \
  --region "$REGION" \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 2 \
  --timeout 1800 \
  --session-affinity \
  --no-cpu-throttling
