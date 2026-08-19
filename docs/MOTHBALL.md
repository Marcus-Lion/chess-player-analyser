# Mothball notes — Chess Player Analyser

Repository: https://github.com/Marcus-Lion/chess-player-analyser

Reference deployment: https://chess-player-analyser-859165106671.us-central1.run.app/

## Purpose

This repository is being placed into mothball status: the code stays available as
open source, but the project should be treated as low-activity / read-only
unless someone explicitly reopens maintenance.

The goal is to preserve:

- the source code,
- the deployment and operational context,
- the data model and environment assumptions,
- and the minimum information needed to restart the app later.

## What this project is

Chess Player Analyser is a Python/FastAPI application with two main paths:

- analysis of public Chess.com game archives for a username, and
- a self-play engine / tuning workflow that stores results in Neo4j.

The repo also includes a native Rust engine under `engine/`, Cloud Run
deployment scripts, and documentation for the Neo4j-backed persistence layer.

More concretely, the current feature set includes:

- reading a public Chess.com username's games and archives,
- building player/game analytics from those archives,
- browsing individual games move by move,
- running self-play between local engine profiles,
- optionally playing the local engine against Lichess AI,
- and a reinforcement-learning training path under `app/rl/`.

The Lichess mode is the remote-opponent path currently implemented in code; if
you want the notes to say "Stockfish-backed" explicitly, that should be treated
as a descriptive label for Lichess AI rather than a separate integration.

## What to preserve before fully retiring anything

If the live service is still considered important, preserve these first:

- a final source backup of the repository,
- any `.env` values or secret material needed to recreate the deployment,
- Neo4j data exports or a full database backup,
- Cloud Run service configuration,
- DNS / domain / routing details if any custom routing exists,
- and a note of any external dependencies that are not recreated from source.

## Operational dependencies

The codebase currently assumes or documents the following runtime pieces:

- Python application runtime via `uv`.
- Native Rust chess engine build under `engine/`.
- Optional Neo4j persistence for human-game export and self-play storage.
- Cloud Run deployment support via the included Dockerfile and scripts.
- External Chess.com API access for fetching public archives.

If any of those external systems disappear, the app may still build, but some
features will no longer work as originally designed.

## Recommended mothballing steps

1. Mark the repository as archived or otherwise read-only in source control.
2. Stop any scheduled deployments, tuning runs, or background jobs.
3. Disable the live demo if ongoing access is no longer desired.
4. Snapshot or export any remaining Neo4j data.
5. Preserve the current deployment config, including Cloud Run settings.
6. Keep this repository and the generated docs available for future recovery.
7. Record the last known working commit or tag before changes stop.

## Reactivation checklist

If the project is revived later, the minimum recovery path is likely:

1. Restore the repo at the last known good commit.
2. Recreate the environment variables and service credentials.
3. Rebuild the Rust engine and Python dependencies.
4. Restore Neo4j if self-play history matters.
5. Redeploy the container and verify the public analysis flow.

## Notes on scope

This project is open source under the MIT License. The mothball state should
not remove the code or documentation; it should make the project easier to
understand, preserve, and restart without guessing at old deployment details.

Thanks to Marc Deveaux for developing this project.
