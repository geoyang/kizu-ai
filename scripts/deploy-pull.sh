#!/bin/bash
# Pull-based deploy script for Kizu-AI.
# Run via cron on the NAS to auto-deploy new images from GHCR.
#
# Setup (one-time on NAS):
#   1. Copy this script to $DEPLOY_PATH/deploy-pull.sh
#   2. chmod +x deploy-pull.sh
#   3. Add cron entry:  */5 * * * * /path/to/deploy-pull.sh >> /var/log/kizu-deploy.log 2>&1
#
# Requires: docker login ghcr.io (already done)

set -euo pipefail

DEPLOY_PATH="${DEPLOY_PATH:-$(cd "$(dirname "$0")" && pwd)}"
COMPOSE_FILE="$DEPLOY_PATH/docker-compose.prod.yml"
IMAGE="ghcr.io/geoyang/kizu-ai:${IMAGE_TAG:-latest}"

cd "$DEPLOY_PATH"

# Pull latest image and capture output
PULL_OUTPUT=$(docker pull "$IMAGE" 2>&1)

# Check if a new image was downloaded
if echo "$PULL_OUTPUT" | grep -q "Downloaded newer image\|Pull complete"; then
    echo "$(date): New image detected, restarting containers..."
    docker compose -f "$COMPOSE_FILE" up -d
    echo "$(date): Deploy complete."
else
    echo "$(date): Image up to date, nothing to do."
fi
