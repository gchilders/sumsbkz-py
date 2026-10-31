#!/bin/bash
# gpu_submit.sh TAG CPUS MEM DEADLINE_SECONDS -- sievesearch args...
# One GPU Job (RTX 4090; user-approved fallbacks L40 / A40 / RTX A6000) running nrp/gpu_run.sh.
# The environment is prebuilt on the PVC, so the GPU starts working within a minute and the
# Job exits (and is deleted by TTL) as soon as the search ends. activeDeadlineSeconds caps it.
set -euo pipefail
TAG=$1; CPUS=$2; MEM=$3; DL=$4; shift 4
[ "${1:-}" = "--" ] && shift
ARGS=""
for a in "$@"; do ARGS="$ARGS, \"$a\""; done
kubectl -n csuf-research apply -f - <<YAML
apiVersion: batch/v1
kind: Job
metadata:
  name: gc-eslp-$TAG
spec:
  backoffLimit: 0
  activeDeadlineSeconds: $DL
  ttlSecondsAfterFinished: 1800
  template:
    metadata:
      labels: {app: gc-eslp-gpu}
    spec:
      restartPolicy: Never
      affinity:
        nodeAffinity:
          requiredDuringSchedulingIgnoredDuringExecution:
            nodeSelectorTerms:
            - matchExpressions:
              - {key: nvidia.com/gpu.product, operator: In, values: [NVIDIA-GeForce-RTX-4090, NVIDIA-L40, NVIDIA-A40, NVIDIA-RTX-A6000]}
          preferredDuringSchedulingIgnoredDuringExecution:
          - weight: 100
            preference:
              matchExpressions:
              - {key: nvidia.com/gpu.product, operator: In, values: [NVIDIA-GeForce-RTX-4090]}
      containers:
      - name: main
        image: nvidia/cuda:12.8.1-devel-ubuntu22.04
        command: ["bash", "/opt/eslp/gpu_run.sh", "$TAG"$ARGS]
        env:
        - {name: THREADS, value: "$CPUS"}
        resources:
          requests: {cpu: "$CPUS", memory: "$MEM", nvidia.com/gpu: 1}
          limits:   {cpu: "$CPUS", memory: "$MEM", nvidia.com/gpu: 1}
        volumeMounts:
        - {mountPath: /data, name: data}
        - {mountPath: /opt/eslp, name: run}
        - {mountPath: /opt/eslp-code, name: code}
      volumes:
      - name: data
        persistentVolumeClaim: {claimName: childers-fac-data}
      - name: run
        configMap: {name: gc-eslp-gpurun}
      - name: code
        configMap: {name: gc-eslp-code}
YAML
