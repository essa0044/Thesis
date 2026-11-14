#!/usr/bin/env bash
# Both search methods one after the other, unattended:  bash sim/run_all.sh   (RUNS per case, default 2)
# Each series copies its results to Windows Downloads\series_auto after every run.
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
RUNS="${RUNS:-2}"
for m in sweep active; do
  SEARCH="$m" RUNS="$RUNS" bash sim/run_series.sh
done
echo "=== both series done"
