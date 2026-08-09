# YouTube Chess Videos Workspace

Workspace for collecting and analysing chess games and commentary from YouTube
videos under the Marcus Lion project.

## Workspace layout

- `config.example.yaml` — source and analysis settings to copy into a local config
- `data/raw/` — downloaded video or audio files (ignored by Cloud Build)
- `data/transcripts/` — local transcription output (ignored by Cloud Build)
- `data/processed/` — normalized metadata and intermediate files (ignored by Cloud Build)
- `data/games/` — extracted PGN files and game metadata
- `data/analysis/` — generated analysis artifacts (ignored by Cloud Build)
- `reports/` — curated reports; generated reports are ignored
- `notebooks/` — exploratory analysis notebooks

## Suggested workflow

1. Add one video per entry in `data/games/videos.csv`.
2. Store downloaded media locally in `data/raw/` and transcripts in `data/transcripts/`.
3. Extract or enter PGNs in `data/games/`.
4. Run chess-engine analysis and keep reproducible summaries in `reports/`.

Raw media, transcripts, intermediate data, and generated analysis are local
working data and are excluded from Cloud Build by the repository's
`.gcloudignore`.
