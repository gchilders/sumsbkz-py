#!/bin/bash
# submit.sh TAG CPUS MEM [VALIDATE] -- sievesearch args...
# Creates a CPU-only Job gc-eslp-TAG in csuf-research running nrp/run.sh (from ConfigMap gc-eslp-run).
# No GPU is requested. Results land in /data/eslp/results/TAG.{txt,log} on the PVC.
set -euo pipefail
TAG=$1; CPUS=$2; MEM=$3; shift 3
VALIDATE=0
if [ "${1:-}" = "VALIDATE" ]; then VALIDATE=1; shift; fi
[ "${1:-}" = "--" ] && shift
ARGS=""
for a in "$@"; do ARGS="$ARGS, \"$a\""; done
kubectl -n csuf-research apply -f - <<EOF
apiVersion: batch/v1
kind: Job
metadata:
  name: gc-eslp-$TAG
spec:
  backoffLimit: 0
  ttlSecondsAfterFinished: 3600
  template:
    metadata:
      labels: {app: gc-eslp}
    spec:
      restartPolicy: Never
      containers:
      - name: main
        image: python:3.12-bookworm
        command: ["bash", "/opt/eslp/run.sh", "$TAG"$ARGS]
        env:
        - {name: THREADS, value: "$CPUS"}
        - {name: VALIDATE, value: "$VALIDATE"}
        resources:
          requests: {cpu: "$CPUS", memory: "$MEM"}
          limits:   {cpu: "$CPUS", memory: "$MEM"}
        volumeMounts:
        - {mountPath: /data, name: data}
        - {mountPath: /opt/eslp, name: run}
        - {mountPath: /opt/eslp-code, name: code}
      volumes:
      - name: data
        persistentVolumeClaim: {claimName: childers-fac-data}
      - name: run
        configMap: {name: gc-eslp-run}
      - name: code
        configMap: {name: gc-eslp-code, optional: true}
EOF
