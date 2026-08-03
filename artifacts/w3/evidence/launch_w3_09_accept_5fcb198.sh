#!/usr/bin/env bash
set -euo pipefail

LAP=/home/yangsen/osx_ur/catkin_ws/src/osx_vla/externals/lap
EVID="${LAP}/artifacts/w3/evidence"
LOG="${EVID}/w3-09-canonical-accept-5fcb198.txt"
DMON="${EVID}/w3-09-gpu-dmon-5fcb198.csv"
EXEC="$(git -C "${LAP}" rev-parse HEAD)"
START="$(date -Iseconds)"

cd "${LAP}"
mkdir -p "${EVID}"

# Stop duplicate dmon from failed prior launches.
pkill -f 'nvidia-smi dmon -s pucvmet -d 30 -o DT' 2>/dev/null || true
sleep 1

{
  echo "START ${START}"
  echo "LAP_HEAD ${EXEC}"
  echo "HOST $(hostname)"
  echo "REPAIR W3-09-R1..R4 fresh canonical accept after sampler/identity/atomic fix"
  nvidia-smi -L
  ls -l /dev/nvidiactl /dev/nvidia0 /dev/nvidia1
  "${LAP}/.venv/bin/python" -c "import torch; print('cuda', torch.cuda.is_available(), torch.cuda.device_count(), torch.__version__)"
} > "${LOG}" 2>&1

nvidia-smi dmon -s pucvmet -d 30 -o DT > "${DMON}" 2>&1 &
echo $! > "${EVID}/w3-09-dmon-5fcb198.pid"

nohup "${LAP}/.venv/bin/python" "${LAP}/scripts/w3_cover.py" \
  --mode accept \
  --w2-root /home/yangsen/osx_ur/catkin_ws/src/osx_vla/.w2_canonical_export_v3 \
  --bridge-artifact /home/yangsen/cover-vla/bridge_verifier/cover_verifier_bridge.pt \
  --audit-manifest "${LAP}/artifacts/w3/bridge_audit_manifest.json" \
  --protocol-dir "${LAP}/artifacts/w3/protocol" \
  --output-root "${LAP}/artifacts/w3/canonical_acceptance_v1" \
  >> "${LOG}" 2>&1 &
echo $! > "${EVID}/w3-09-accept-5fcb198.pid"

echo "ACCEPT_PID=$(cat "${EVID}/w3-09-accept-5fcb198.pid")"
echo "DMON_PID=$(cat "${EVID}/w3-09-dmon-5fcb198.pid")"
echo "LOG=${LOG}"
echo "EXEC=${EXEC}"
