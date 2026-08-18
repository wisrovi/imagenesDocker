#!/usr/bin/env bash
set -euo pipefail

# -------------------------------------------------------------------
# docker_auth_setup.sh – Create an authenticated session for agy
# -------------------------------------------------------------------
# This script runs the agy Docker image interactively, lets the user
# perform the OAuth login once, and then copies the resulting config
# files back to the host ($HOME/.config/agy-edc).
# -------------------------------------------------------------------

# Temporary directory that will be mounted into the container.
TMP_AUTH_DIR="$(pwd)/agy-auth-tmp"
mkdir -p "$TMP_AUTH_DIR"

echo "=================================================================="
echo "Este script abrirá la UI de agy dentro del contenedor Docker."
echo "Sigue los pasos para iniciar sesión (Google OAuth u otro método)."
echo "Cuando veas el mensaje de éxito, cierra la UI (Ctrl‑C o escribe 'exit')."
echo "=================================================================="
read -rp "Presiona ENTER para iniciar la autenticación..."

# Run the container interactively with the temporary auth directory mounted.
# -it forces a pseudo‑TTY so the Bubble Tea UI works.
# The container will write its config into /root/.config/agy-edc which is
# mapped to $TMP_AUTH_DIR on the host.

docker run --rm -it \
  -v "$TMP_AUTH_DIR:/root/.config/agy-edc:rw" \
  wisrovi/agy-edc:v1.1.0 agy

# After the container exits, copy the generated config to the host's real
# config directory (~/.config/agy-edc). Create it if it does not exist.
HOST_AUTH_DIR="${HOME}/.config/agy-edc"
mkdir -p "$HOST_AUTH_DIR"

echo "Copiando la configuración autenticada al directorio del host..."
rsync -a --delete "$TMP_AUTH_DIR/" "$HOST_AUTH_DIR/"

# Clean up temporary directory.
rm -rf "$TMP_AUTH_DIR"

echo "✅ Configuración de autenticación guardada en $HOST_AUTH_DIR"

echo "Ahora puedes ejecutar ./docker_test.sh sin que solicite login."
