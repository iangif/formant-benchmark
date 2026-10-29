#!/usr/bin/env bash
# Run experiments one after another, logging each to logs/experiments/<name>.log.
#
#   scripts/run_experiments.sh                   # every configs/experiments/*.yaml except template
#   scripts/run_experiments.sh vtr-praat_burg …  # only the named experiments
#
# A failing experiment is logged and the queue moves on. Extra flags for
# `experiment run` (e.g. --resume) can be passed through EXPERIMENT_FLAGS.
# Run from the repository root.

set -u

if [ "$#" -gt 0 ]; then
  experiments=("$@")
else
  experiments=()
  for path in configs/experiments/*.yaml; do
    name=$(basename "$path" .yaml)
    [ "$name" = "template" ] || experiments+=("$name")
  done
fi

log_dir=logs/experiments
mkdir -p "$log_dir"
summary="$log_dir/summary-$(date +%Y%m%d-%H%M%S).txt"
failed=0

for name in "${experiments[@]}"; do
  echo "[$(date '+%F %T')] start $name" | tee -a "$summary"
  # shellcheck disable=SC2086
  if uv run formant-benchmark experiment run "$name" ${EXPERIMENT_FLAGS:-} >"$log_dir/$name.log" 2>&1; then
    status=ok
  else
    status="FAILED (exit $?)"
    failed=$((failed + 1))
  fi
  echo "[$(date '+%F %T')] $status $name" | tee -a "$summary"
done

echo "$((${#experiments[@]} - failed))/${#experiments[@]} experiments succeeded; summary in $summary" | tee -a "$summary"
[ "$failed" -eq 0 ]
