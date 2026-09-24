#!/usr/bin/env bash
set -euo pipefail

image_tar="${1:-alliage-hacka.tar}"
test -f "$image_tar" || { echo "Imagem offline não encontrada: $image_tar"; exit 1; }
docker load --input "$image_tar"
docker compose -f docker-compose.yml up --no-build
