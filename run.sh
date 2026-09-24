#!/usr/bin/env bash
set -euo pipefail

command -v docker >/dev/null 2>&1 || { echo "Docker não está instalado."; exit 1; }
docker info >/dev/null 2>&1 || { echo "O daemon do Docker não está disponível."; exit 1; }

compose=(docker compose -f docker-compose.yml)
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
  compose+=( -f docker-compose.gpu.yml )
  echo "GPU NVIDIA detectada; iniciando com suporte a GPU."
else
  echo "GPU NVIDIA não disponível; iniciando em CPU."
fi
"${compose[@]}" up --build
