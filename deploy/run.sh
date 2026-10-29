#!/bin/sh
set -eu
TASTETRAIL_REPO_DIR=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$TASTETRAIL_REPO_DIR"
export PYTHONPATH="$TASTETRAIL_REPO_DIR:$TASTETRAIL_REPO_DIR/work/culture_v13:$TASTETRAIL_REPO_DIR/work/culture_v11:$TASTETRAIL_REPO_DIR/work/culture_v11/src${PYTHONPATH:+:$PYTHONPATH}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-2}"
if [ "${TASTETRAIL_STARTUP_CHECK:-1}" = "1" ]; then
  "${TASTETRAIL_PYTHON:-python}" deploy/check_runtime.py
fi
# One worker is intentional: admission counters and inference locks are local.
exec "${TASTETRAIL_PYTHON:-python}" -m uvicorn deploy.gateway:app \
  --host "${TASTETRAIL_HOST:-127.0.0.1}" --port "${PORT:-8773}" \
  --workers 1 --no-access-log --no-proxy-headers
