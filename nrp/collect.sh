#!/bin/bash
# collect.sh TAG...  -- save the Job's log to results/nrp/TAG.log (must contain "done"), then delete the Job.
cd "$(dirname "$0")/.."
for TAG in "$@"; do
  for t in 1 2 3 4 5; do
    kubectl -n csuf-research logs job/gc-eslp-$TAG > results/nrp/$TAG.log 2>/dev/null && grep -q "== $TAG done" results/nrp/$TAG.log && break
    sleep 10
  done
  if grep -q "== $TAG done" results/nrp/$TAG.log; then
    kubectl -n csuf-research delete job gc-eslp-$TAG 2>&1 | grep -v memcache
  else
    echo "$TAG: log incomplete, Job kept"
  fi
done
