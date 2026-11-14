#!/bin/bash
# Keep the transformer weights of every EVERY-th step, but only the newest KEEP of them per run
# (KEEP defaults to 4, the user's choice on 2026-11-12; KEEP=0 keeps all).
# The training job saves a resume checkpoint every 2 steps and deletes all but the newest two, so this
# loop copies checkpoints/step-XXXXXX/transformer to milestones/step-XXXXXX/transformer inside the
# bucket (server-side copy, nothing is downloaded) before the training job prunes it.
#   setsid nohup ./archiver.sh 10 <job> [<job> ...] > archiver.log 2>&1 &
EVERY="$1"; shift
KEEP="${KEEP:-4}"
G="${GCS_ROOT:-gs://prism-training-apac/jiaming/npo-sana}"
export RAY_API_SERVER_ADDRESS="${RAY_API_SERVER_ADDRESS:-http://127.0.0.1:8265}"
while true; do
  active=0
  for J in "$@"; do
    cks=$(gcloud storage ls "$G/$J/checkpoints/" 2>/dev/null | grep -o "step-[0-9]*" | sort -u)
    # A restart after a spot preemption leaves the pre-restart resume checkpoints behind in the bucket
    # (the new workers only know about the one they resumed from). Remove everything older than the
    # newest three resume checkpoints so they do not pile up or get archived again.
    nck=$(echo "$cks" | grep -c .)
    if [ "$nck" -gt 3 ]; then
      for old in $(echo "$cks" | head -n $((nck - 3))); do
        gcloud storage rm -r --quiet "$G/$J/checkpoints/$old" >/dev/null 2>&1 && echo "$(date '+%m/%d %H:%M:%S') removed stale resume checkpoint $J $old"
      done
      cks=$(echo "$cks" | tail -n 3)
    fi
    # never archive a step that is older than the milestones already kept (it would be pruned at once)
    oldest_kept=0
    if [ "$KEEP" -gt 0 ]; then
      ms=$(gcloud storage ls "$G/$J/milestones/" 2>/dev/null | grep -o "step-[0-9]*" | sort -u)
      if [ "$(echo "$ms" | grep -c .)" -ge "$KEEP" ]; then
        o=$(echo "$ms" | tail -n "$KEEP" | head -1); oldest_kept=$((10#${o#step-}))
      fi
    fi
    for d in $cks; do
      step=$((10#${d#step-}))
      [ $((step % EVERY)) -eq 0 ] || continue
      [ "$step" -gt "$oldest_kept" ] || [ "$oldest_kept" -eq 0 ] || continue
      # the checkpoint is complete only once latest.json points at it or at a later one
      latest=$(gcloud storage cat "$G/$J/checkpoints/latest.json" 2>/dev/null | grep -o '"step": [0-9]*' | grep -o "[0-9]*")
      [ -n "$latest" ] && [ "$latest" -ge "$step" ] || continue
      if ! gcloud storage ls "$G/$J/milestones/$d/transformer/config.json" >/dev/null 2>&1; then
        if gcloud storage cp -r --quiet "$G/$J/checkpoints/$d/transformer" "$G/$J/milestones/$d/" >/dev/null 2>&1 \
           && gcloud storage ls "$G/$J/milestones/$d/transformer/diffusion_pytorch_model.safetensors" >/dev/null 2>&1; then
          echo "$(date '+%m/%d %H:%M:%S') archived $J $d"
        else
          echo "$(date '+%m/%d %H:%M:%S') FAILED to archive $J $d (will retry)"
          gcloud storage rm -r --quiet "$G/$J/milestones/$d" >/dev/null 2>&1
        fi
      fi
    done
    # prune: keep only the newest $KEEP milestones of this run
    if [ "$KEEP" -gt 0 ]; then
      all=$(gcloud storage ls "$G/$J/milestones/" 2>/dev/null | grep -o "step-[0-9]*" | sort -u)
      n=$(echo "$all" | grep -c .)
      if [ "$n" -gt "$KEEP" ]; then
        for old in $(echo "$all" | head -n $((n - KEEP))); do
          gcloud storage rm -r --quiet "$G/$J/milestones/$old" >/dev/null 2>&1 && echo "$(date '+%m/%d %H:%M:%S') pruned $J $old (keeping newest $KEEP)"
        done
      fi
    fi
    st=$(curl -s --max-time 20 "http://127.0.0.1:8265/api/jobs/$J" | python3 -c 'import sys,json; print(json.load(sys.stdin).get("status",""))' 2>/dev/null)
    case "$st" in SUCCEEDED|FAILED|STOPPED) ;; *) active=1;; esac
  done
  [ "$active" = 0 ] && { echo "$(date '+%m/%d %H:%M:%S') all jobs finished, archiver exits"; exit 0; }
  sleep 240
done
