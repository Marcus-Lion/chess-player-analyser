#!/usr/bin/env bash
#
# One-shot deployment script for the Marcus Lion Chess Player Analyser
# on a fresh Ubuntu (22.04/24.04) Hostinger VPS.
#
# Usage (run as root or with sudo, on the VPS):
#   DOMAIN=example.com EMAIL=you@example.com ./deploy.sh
#
# Optional environment variables:
#   DOMAIN       Domain to serve + request a TLS certificate for (skips TLS if unset)
#   EMAIL        Email for Let's Encrypt registration (required if DOMAIN is set)
#   APP_USER     Linux user that owns/runs the app        (default: current sudo user or "chess")
#   APP_PORT     Local port uvicorn listens on            (default: 8000)
#   REPO_URL     Git repository to clone                  (default: project's GitHub repo)
#   APP_DIR      Where to install the app                 (default: /home/$APP_USER/chess-player-analyser)
#
set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
APP_USER="${APP_USER:-${SUDO_USER:-chess}}"
APP_PORT="${APP_PORT:-8000}"
REPO_URL="${REPO_URL:-https://github.com/marcus-lion/chess-player-analyser.git}"
APP_DIR="${APP_DIR:-/home/${APP_USER}/chess-player-analyser}"
SERVICE_NAME="chess-analyser"
DOMAIN="${DOMAIN:-}"
EMAIL="${EMAIL:-}"

log() { printf '\n\033[1;32m==> %s\033[0m\n' "$1"; }

if [[ $EUID -ne 0 ]]; then
  echo "Please run this script as root (or with sudo)." >&2
  exit 1
fi

# ---------------------------------------------------------------------------
# 1. Ensure the app user exists
# ---------------------------------------------------------------------------
if ! id -u "$APP_USER" >/dev/null 2>&1; then
  log "Creating system user '$APP_USER'"
  useradd --create-home --shell /bin/bash "$APP_USER"
fi

# ---------------------------------------------------------------------------
# 2. System packages
# ---------------------------------------------------------------------------
log "Installing system packages"
export DEBIAN_FRONTEND=noninteractive
apt-get update -y
apt-get upgrade -y
apt-get install -y nginx git curl

# ---------------------------------------------------------------------------
# 3. Install uv for the app user
# ---------------------------------------------------------------------------
UV_BIN="/home/${APP_USER}/.local/bin/uv"
if [[ ! -x "$UV_BIN" ]]; then
  log "Installing uv for '$APP_USER'"
  sudo -u "$APP_USER" bash -c 'curl -LsSf https://astral.sh/uv/install.sh | sh'
fi

# ---------------------------------------------------------------------------
# 4. Clone / update the repository and sync dependencies
# ---------------------------------------------------------------------------
if [[ -d "$APP_DIR/.git" ]]; then
  log "Updating existing checkout at $APP_DIR"
  sudo -u "$APP_USER" git -C "$APP_DIR" pull --ff-only
else
  log "Cloning repository into $APP_DIR"
  sudo -u "$APP_USER" git clone "$REPO_URL" "$APP_DIR"
fi

log "Syncing Python dependencies with uv"
sudo -u "$APP_USER" bash -c "cd '$APP_DIR' && '$UV_BIN' sync"

# ---------------------------------------------------------------------------
# 5. systemd service
# ---------------------------------------------------------------------------
log "Writing systemd unit /etc/systemd/system/${SERVICE_NAME}.service"
cat >"/etc/systemd/system/${SERVICE_NAME}.service" <<EOF
[Unit]
Description=Marcus Lion Chess Player Analyser
After=network.target

[Service]
User=${APP_USER}
WorkingDirectory=${APP_DIR}
Environment="PATH=${APP_DIR}/.venv/bin"
ExecStart=${APP_DIR}/.venv/bin/uvicorn app.main:app --host 127.0.0.1 --port ${APP_PORT}
Restart=always

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable --now "${SERVICE_NAME}"
systemctl restart "${SERVICE_NAME}"

# ---------------------------------------------------------------------------
# 6. Nginx reverse proxy
# ---------------------------------------------------------------------------
SERVER_NAME="${DOMAIN:-_}"
log "Configuring Nginx reverse proxy (server_name: ${SERVER_NAME})"
cat >"/etc/nginx/sites-available/${SERVICE_NAME}" <<EOF
server {
    listen 80;
    server_name ${SERVER_NAME};

    location / {
        proxy_pass http://127.0.0.1:${APP_PORT};
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
EOF

ln -sf "/etc/nginx/sites-available/${SERVICE_NAME}" "/etc/nginx/sites-enabled/${SERVICE_NAME}"
rm -f /etc/nginx/sites-enabled/default
nginx -t
systemctl reload nginx

# ---------------------------------------------------------------------------
# 7. Firewall (only if ufw is active)
# ---------------------------------------------------------------------------
if command -v ufw >/dev/null 2>&1 && ufw status | grep -q "Status: active"; then
  log "Opening firewall ports"
  ufw allow 'Nginx Full'
  ufw allow OpenSSH
fi

# ---------------------------------------------------------------------------
# 8. TLS certificate (optional)
# ---------------------------------------------------------------------------
if [[ -n "$DOMAIN" ]]; then
  if [[ -z "$EMAIL" ]]; then
    echo "DOMAIN is set but EMAIL is empty; skipping TLS. Re-run with EMAIL=... to enable HTTPS." >&2
  else
    log "Requesting Let's Encrypt certificate for $DOMAIN"
    apt-get install -y certbot python3-certbot-nginx
    certbot --nginx --non-interactive --agree-tos -m "$EMAIL" -d "$DOMAIN" || \
      echo "certbot failed (DNS not pointing here yet?). You can re-run: certbot --nginx -d $DOMAIN"
  fi
fi

log "Deployment complete!"
echo "Service:  systemctl status ${SERVICE_NAME}"
echo "Logs:     journalctl -u ${SERVICE_NAME} -f"
if [[ -n "$DOMAIN" ]]; then
  echo "Visit:    http://${DOMAIN}  (https:// if TLS succeeded)"
else
  echo "Visit:    http://YOUR_SERVER_IP"
fi
