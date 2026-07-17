# Deployment

## Hostinger VPS deployment

### Automated (recommended)

SSH into your VPS as root and run the bundled `deploy.sh`. It installs
system packages and `uv`, clones the repo, syncs dependencies, and sets up
`systemd` + Nginx (and optionally HTTPS via Let's Encrypt):

```bash
curl -LsSf https://raw.githubusercontent.com/marcus-lion/chess-player-analyser/main/deploy.sh -o deploy.sh
chmod +x deploy.sh
sudo DOMAIN=yourdomain.com EMAIL=you@yourdomain.com ./deploy.sh
```

Omit `DOMAIN`/`EMAIL` to deploy over plain HTTP on the server IP.

### Manual

```bash
sudo apt update && sudo apt upgrade -y
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

git clone https://github.com/marcus-lion/chess-player-analyser.git
cd chess-player-analyser

uv sync
uv run uvicorn app.main:app --host 127.0.0.1 --port 8134
```

For production, run with `systemd` and put Nginx in front as a reverse proxy.

## GCP Cloud Run deployment

The app also runs on Cloud Run, built from the repo's `Dockerfile` via Cloud
Build (no Artifact Registry setup needed):

```bash
gcloud run deploy chess-player-analyser \
  --source . \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 2Gi --cpu 2 --timeout 1800 \
  --session-affinity \
  --no-cpu-throttling
```

Live deployment: project `chess-player-502601`, region `us-central1`, service
`chess-player-analyser` at
https://chess-player-analyser-859165106671.us-central1.run.app (public, no
auth -- matches the plain-HTTP Hostinger setup above). Sized at 2 vCPU / 2Gi
memory with a 30-minute request timeout (up from the Cloud Run defaults of 1
vCPU / 512Mi / 5 minutes) after self-play runs were getting OOM-killed --
Cloud Logging showed repeated `Memory limit of 512 MiB exceeded` errors that
took the WebSocket connection down mid-run. The underlying trigger was
`app/self_play.py`'s default worker count reading the *host* machine's CPU
count instead of what the container was actually allocated, over-spawning
worker processes; it now uses `os.process_cpu_count()`, which respects the
container's cgroup CPU quota.

### Private Neo4j on GCP

Self-play (and optional `/analyse` export) needs Neo4j. In production it runs
on a Compute Engine VM with **no external IP**, reachable only from Cloud Run
over a private VPC connection -- never exposed to the public internet:

- VM `neo4j-server` (e2-small, `us-central1-a`, Container-Optimized OS,
  `--no-address`) running the `neo4j:5` container.
- Firewall `allow-neo4j-from-vpc-connector`: tcp:7687 (bolt) only, source
  restricted to the VPC connector's subnet (`10.8.0.0/28`) -- never
  `0.0.0.0/0`.
- Serverless VPC Access connector `cloudrun-to-vpc` (`us-central1`) lets Cloud
  Run reach the VM's internal IP.
- Cloud Router/NAT (`nat-router`/`nat-config`) gives the VM outbound-only
  internet access (needed to pull container images) without an external IP or
  any inbound exposure.
- Cloud Run is wired to it with `--vpc-connector cloudrun-to-vpc` plus the
  same `NEO4J_ENABLED`/`NEO4J_URI`/`NEO4J_USER`/`NEO4J_PASSWORD` env vars as
  the [Neo4j option](README.md#neo4j-option) below, with `NEO4J_URI` pointing at the
  VM's internal IP instead of `localhost`.

Neo4j Browser isn't exposed publicly; admin access is via an SSH tunnel:

```bash
gcloud compute ssh neo4j-server --zone=us-central1-a --tunnel-through-iap -- -L 7474:localhost:7474
# then open http://localhost:7474 locally
```

Data lives on the VM's boot disk rather than a separate persistent disk, so
deleting/recreating the VM loses it -- the same ephemeral-storage tradeoff
already accepted for Cloud Run's local file cache. Unlike Cloud Run, none of
this scales to zero: budget roughly $20-25/month ongoing for the VM,
connector, and NAT.

### Cost

- **Cloud Run** bills per request-second of actual CPU/memory used and
  scales to zero when idle, so the 1→2 vCPU / 512Mi→2Gi resize roughly
  doubles the *active-second* rate (~$0.000025/s → ~$0.000053/s) rather than
  adding a fixed cost. The free tier (180k vCPU-seconds + 360k GiB-seconds/
  month) covers light/occasional use; the main cost driver is self-play run
  duration now that the request timeout is 1800s instead of 300s -- a ~10
  minute self-play run at 2 vCPU/2Gi costs roughly $0.03.
- **Neo4j VM + connector + NAT** (above) is the real fixed cost: ~$20-25/month,
  billed continuously since none of it scales to zero, independent of how
  much the app is actually used.
