#!/usr/bin/env bash
# Serves the Nordic AI Cup 2026 case-2 (drone-flyby) endpoint, using the exact
# configuration that scored 0.3526 on the validation board (5-run spread 0.010)
# and 0.333 mean over 10 fresh board attempts on 2026-09-20.
#
#   pip install -r requirements.txt
#   ./run.sh
#
# Then verify: curl http://localhost:9053/api
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

export DRONE_WEIGHTS="weights/drone_v5_v8.pt"
export DRONE_WEIGHTS2="weights/synth_v4_e4.pt"
export DRONE_WEIGHTS2_CLASSES="condor,spacecraft,medium_plane,jammer,ta-ta,large_launcher"
export DRONE_BOX_SCALE="hangar=0.94,helicopter=1.3,jet_plane=1.15,large_tower=1.34,medium_launcher=1.59,mine_roller=0.84,small_launcher=1.04,small_plane=0.99,small_tower=1.08,tank=0.83,condor=1.0,spacecraft=1.0,medium_plane=1.0,jammer=1.0,ta-ta=1.0,large_launcher=1.0"
export DRONE_POLICY="l1band"
export DRONE_PORT="${DRONE_PORT:-9053}"

python3 drone_server.py
