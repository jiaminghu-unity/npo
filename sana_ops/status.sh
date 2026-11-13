#!/bin/bash
# One-screen health check of the Sana runs.
export RAY_API_SERVER_ADDRESS=http://127.0.0.1:8265
GCS=gs://prism-training-apac/jiaming/npo-sana
echo "== $(date '+%m/%d %H:%M:%S') nodes"
gcloud compute instances list --filter="labels.project=npo-sana" --format="table[no-heading](name,zone.basename(),status,labels.pool,creationTimestamp.date('%m/%d %H:%M'))" 2>&1
# Runs: two NPO runs (pools a, b) and, since 2026-11-13 00:27 UTC, the Online-DPO avg-rank control (pool d).
# The offline DPO run (pool c) finished on 2026-11-12 18:05 UTC (early stop at step 420); it is not listed any more.
echo "== keepers (expect 2: pools a+b, pool d)"; echo "keeper processes: $(pgrep -f "^/bin/bash ./keeper.s[h] [0-9]" | wc -l)"
pgrep -f "keeper.sh 4 a:" >/dev/null || echo "KEEPER a/b NOT RUNNING"
pgrep -f "keeper.sh 4 d:" >/dev/null || echo "KEEPER d (online DPO) NOT RUNNING"
tail -2 ~/npo_sana_dev/nodes/keeper.log | cut -c1-160
tail -1 ~/npo_sana_dev/nodes/keeper_d.log | cut -c1-160
echo "== archivers (expect 2) (keep every 10th step's weights in <run>/milestones/)"
pgrep -f "archiver.sh 10 sana-newnash" >/dev/null || echo "ARCHIVER (npo runs) NOT RUNNING"
pgrep -f "archiver.sh 10 sana-onlinedpo" >/dev/null || echo "ARCHIVER (online DPO) NOT RUNNING"
tail -2 ~/npo_sana_dev/archiver.log 2>/dev/null | cut -c1-160; tail -1 ~/npo_sana_dev/archiver_odpo.log 2>/dev/null | cut -c1-160
for J in sana-newnash-randjudge-v1 sana-newnash-avgrank-v1 sana-onlinedpo-avgrank-v1; do
  echo "== $J: $(curl -s --max-time 20 "http://127.0.0.1:8265/api/jobs/$J" | python3 -c 'import sys,json; print(json.load(sys.stdin).get("status",""))' 2>/dev/null)"
  L=$(ray job logs $J 2>&1 | sed 's/^.*ip=\([0-9.]*\))[^ ]* /\1 /')
  echo "$L" | grep -a "sana_ray_job\] \(attempt\|worker\|training\|too many\)" | tail -3 | cut -c1-200
  echo "restarts so far: $(echo "$L" | grep -ac 'restarting the group')"
  echo "$L" | grep -av "autoscaler" | grep -a "Traceback\|Error\|non-finite\|early stop" | tail -3 | cut -c1-200
  echo "$L" | grep -a "\[step " | tail -2 | cut -c1-330
  echo "  milestones kept: $(gcloud storage ls $GCS/$J/milestones/ 2>/dev/null | grep -o 'step-[0-9]*' | sed 's/step-0*//' | tr '\n' ' ')"
  gcloud storage cat $GCS/$J/eval.jsonl 2>/dev/null | python3 -I -c "
import sys, json
rows=[json.loads(l) for l in sys.stdin if l.strip()]
for r in rows[:1]+rows[-4:] if len(rows)>5 else rows:
    print('  eval step %4d gain %+.4f%s | ' % (r['step'], r.get('gain',0), ' *best' if r.get('is_best') else '') + ' '.join('%s=%.4f'%(k,r[k]) for k in ('pickscore','hps','clip','aes','im')))
"
done
