# Firebase upgrade path

The MVP uses local file caching because it is simple and cheap on a VPS.

Firebase can be added in phases:

## Phase 1
Use Firebase Authentication for login.

## Phase 2
Save analysis metadata in Firestore:

```text
users/{uid}
analyses/{username}
analyses/{username}/runs/{run_id}
```

## Phase 3
Store raw PGN files in Firebase Cloud Storage.

## Phase 4
Move the frontend to Firebase Hosting and keep FastAPI as an API backend on Hostinger, Cloud Run, or another Python host.

Suggested package:

```bash
pip install firebase-admin
```
