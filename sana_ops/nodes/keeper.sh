#!/bin/bash
# Keep the dedicated spot A100 nodes alive while the Ray jobs are active.
#   setsid nohup ./keeper.sh 4 a:<job-a> b:<job-b> > keeper.log 2>&1 &
# For each "pool:job" pair, N nodes of that pool are kept while the job is RUNNING/PENDING.
# When a job has finished (SUCCEEDED / FAILED / STOPPED), the nodes of its pool are deleted.
# The script exits when all jobs have finished.
N="$1"; shift
DIR="$(cd "$(dirname "$0")" && pwd)"
export RAY_API_SERVER_ADDRESS="${RAY_API_SERVER_ADDRESS:-http://127.0.0.1:8265}"
while true; do
  active=0
  for pj in "$@"; do
    pool="${pj%%:*}"; job="${pj#*:}"
    # Job status from the REST API (exact field). `ray job status` prints "Job '<id>' succeeded" in lower case,
    # which the old grep for SUCCEEDED never matched, so finished pools kept their nodes (found 2026-11-12).
    # An unreachable API gives an empty status, which counts as active: nodes are never deleted by mistake.
    st=$(curl -s --max-time 20 "$RAY_API_SERVER_ADDRESS/api/jobs/$job" | python3 -c 'import sys,json; print(json.load(sys.stdin).get("status",""))' 2>/dev/null)
    F="labels.project=npo-sana AND labels.pool=$pool"
    case "$st" in
      SUCCEEDED|FAILED|STOPPED)
        gcloud compute instances list --filter="$F" --format="value(name,zone.basename())" 2>/dev/null | while read name zone; do
          echo "$(date '+%m/%d %H:%M:%S') job $job is $st -> deleting $name"
          gcloud compute instances delete "$name" --zone "$zone" --quiet >/dev/null 2>&1
        done ;;
      *)
        active=1
        names=$(gcloud compute instances list --filter="$F AND status:(RUNNING PROVISIONING STAGING)" --format="value(name)" 2>/dev/null) || continue
        cnt=$(echo "$names" | grep -c .)
        echo "$(date '+%m/%d %H:%M:%S') pool $pool job $job [${st:-unknown}] nodes=$cnt/$N"
        for k in $(seq 1 $((N-cnt))); do
          "$DIR/make_node.sh" "npo-$pool-$(date +%d%H%M%S)-$k" "$pool" 2>&1 | grep "CREATED\|FAILED" &
        done
        wait ;;
    esac
  done
  [ "$active" = 0 ] && { echo "$(date '+%m/%d %H:%M:%S') all jobs finished, keeper exits"; exit 0; }
  sleep 120
done
