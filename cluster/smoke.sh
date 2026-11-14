#!/bin/bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
P=${SFPT_PYTHON:-python}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2} MPLBACKEND=Agg
for case_name in dialanine_vacuum_sfpt tetra_alanine_sfpt dialanine_vacuum_metad tetra_alanine_metad dialanine_water_sfpt dialanine_water_metad rearrangement_sfpt rearrangement_metad; do
 "$P" "$ROOT/run_case.py" "$case_name" --steps 100 --output "$ROOT/smoke/$case_name"
done
