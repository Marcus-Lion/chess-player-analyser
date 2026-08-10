@echo off
setlocal enabledelayedexpansion

REM Deploy the app to Google Cloud Run with the self-play flags enabled.
REM
REM Optional environment variables:
REM   SERVICE_NAME  Cloud Run service name (default: chess-player-analyser)
REM   REGION        Cloud Run region (default: us-central1)

set "SERVICE_NAME=%SERVICE_NAME%"
if not defined SERVICE_NAME set "SERVICE_NAME=chess-player-analyser"

set "REGION=%REGION%"
if not defined REGION set "REGION=us-central1"

gcloud run deploy "%SERVICE_NAME%" ^
  --source . ^
  --region "%REGION%" ^
  --allow-unauthenticated ^
  --memory 2Gi ^
  --cpu 2 ^
  --timeout 1800 ^
  --session-affinity ^
  --no-cpu-throttling

endlocal
