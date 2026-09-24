#!/usr/bin/env bash
set -euo pipefail

command -v docker >/dev/null 2>&1 || { echo "Docker não está instalado."; exit 1; }
docker info >/dev/null 2>&1 || { echo "O daemon do Docker não está disponível."; exit 1; }
docker compose -f docker-compose.yml build
docker compose -f docker-compose.cloud.yml up
