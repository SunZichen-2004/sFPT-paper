#!/bin/bash
set -uo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export OMP_NUM_THREADS=4 MPLBACKEND=Agg PYTHONUNBUFFERED=1 TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
P=${SFPT_PYTHON:-python}
for case_name in dialanine_vacuum_sfpt tetra_alanine_sfpt dialanine_vacuum_metad tetra_alanine_metad dialanine_water_sfpt dialanine_water_metad; do
 "$P" -u "$ROOT/run_case.py" "$case_name" > "$ROOT/logs/${case_name}.log" 2>&1
 echo "$case_name exit=$?"
done
