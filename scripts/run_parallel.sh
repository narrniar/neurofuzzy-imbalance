#!/usr/bin/env bash
# Run one process per dataset setting in parallel, then build the report.
#   bash scripts/run_parallel.sh configs/full.yaml [threads_per_process]
# Each process writes its own results/runs/<name>.<setting>.jsonl, so they never
# share a file; re-running the script resumes from finished fits.
set -euo pipefail
cd "$(dirname "$0")/.."
CONFIG="${1:-configs/full.yaml}"
THREADS="${2:-2}"
NAME=$(.venv/bin/python -c "import yaml,sys; print(yaml.safe_load(open(sys.argv[1]))['name'])" "$CONFIG")
SETTINGS=$(.venv/bin/python -c "import yaml,sys; print(' '.join(yaml.safe_load(open(sys.argv[1]))['settings']))" "$CONFIG")
mkdir -p results/logs
pids=()
for s in $SETTINGS; do
  log="results/logs/${NAME}.${s/:/-}.log"
  echo "starting $s -> $log"
  .venv/bin/python -m nfi run "$CONFIG" --settings "$s" --threads "$THREADS" > "$log" 2>&1 &
  pids+=($!)
done
status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=1
done
if [ $status -ne 0 ]; then
  echo "at least one process failed; see results/logs/${NAME}.*.log" >&2
fi
.venv/bin/python -m nfi report "$CONFIG"
exit $status
