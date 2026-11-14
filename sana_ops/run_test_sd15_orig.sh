#!/bin/bash
# SD1.5 test with the user's ORIGINAL eval.py (user instruction 2026-11-12: SD1.5 models are tested with
# eval.py itself; eval_sana.py is only for the Sana models).
#
#   ./run_test_sd15_orig.sh <tag> <pool> <seed> [<seed> ...]
#
# Models: original SD1.5, official Diffusion-DPO (mhdang/dpo-sd1.5-text2image-v1), official Diffusion-KTO
# (jacklishufan/diffusion-kto); arguments as in eval.sh (batch 16, one seed per call).
# eval.py runs unchanged through orig_eval_check/run_orig_eval.py (it only prepares the environment).
# HPSv3 (the added column) is computed from the images eval.py saved. Several nodes can run different
# seeds of the same tag; the summary is made afterwards from the merged results:
#   gs://prism-training-apac/jiaming/npo-sana/tests/<tag>/results/
set -uo pipefail
TAG="$1"; POOL="$2"; shift 2
SEEDS="$*"
DIR="$(cd "$(dirname "$0")" && pwd)"
G=gs://prism-training-apac/jiaming/npo-sana
KEY="-i $HOME/.ssh/ray_worker_key -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -o ConnectTimeout=20"
OUT="$DIR/tests/$TAG"; mkdir -p "$OUT"
log() { echo "$(date '+%m/%d %H:%M:%S') $*"; }

NODE="npo-test-$POOL-$(date +%d%H%M%S)"
log "creating node $NODE for seeds $SEEDS"
"$DIR/nodes/make_node.sh" "$NODE" "$POOL" || { log "TEST_FAILED: could not get a GPU"; exit 1; }
ZONE=$(gcloud compute instances list --filter="name=$NODE" --format="value(zone.basename())")
IP=$(gcloud compute instances list --filter="name=$NODE" --format="value(networkInterfaces[0].networkIP)")
log "node $NODE ($ZONE, $IP)"
cleanup() { log "deleting node $NODE"; gcloud compute instances delete "$NODE" --zone "$ZONE" --quiet >/dev/null 2>&1; }
trap cleanup EXIT
for i in $(seq 1 60); do ssh $KEY ubuntu@$IP 'echo up' 2>/dev/null | grep -q up && break; sleep 10; done
for i in $(seq 1 90); do ssh $KEY ubuntu@$IP 'sudo grep -q NPO_NODE_READY /var/log/npo-startup.log' 2>/dev/null && break; sleep 10; done
log "node boot script finished"

cat > "$OUT/node_script_$POOL.sh" <<EOF
#!/bin/bash
export HOME=/root TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 TOKENIZERS_PARALLELISM=false
cd /root/npo
bash launchers/sana_setup_env.sh > /root/setup_main.log 2>&1 || { echo MAIN_SETUP_FAILED; exit 1; }
if [ ! -f /root/hpsv3-venv/.ready ]; then
  python3 -m pip install -q --user virtualenv
  python3 -m virtualenv -q /root/hpsv3-venv
  /root/hpsv3-venv/bin/pip install -q --upgrade pip
  /root/hpsv3-venv/bin/pip install -q hpsv3 matplotlib tensorboard > /root/setup_hpsv3.log 2>&1 || { echo HPSV3_SETUP_FAILED; exit 1; }
  touch /root/hpsv3-venv/.ready
fi
T=/root/tests/$TAG; mkdir -p \$T/images \$T/results \$T/logs
PY=/root/sana-venv/bin/python
for s in $SEEDS; do
  rm -rf eval_result_rebuttal
  \$PY /root/run_orig_eval.py --original_model stable-diffusion-v1-5/stable-diffusion-v1-5 \\
    --unet_checkpoints mhdang/dpo-sd1.5-text2image-v1 jacklishufan/diffusion-kto --epochs none none \\
    --names "Orig. SD1.5" "Diffusion-DPO" "Diffusion-KTO" \\
    --device_orig cuda:0 --device_trained cuda:0 --metrics_device cuda:0 --batch_size 16 --seed \$s \\
    --save_dir \$T/images > \$T/logs/evalpy_seed\$s.log 2>&1 || { echo "EVALPY_FAILED seed \$s"; exit 1; }
  \$PY /root/convert_evalpy_csv.py eval_result_rebuttal \$T/results > \$T/logs/convert_seed\$s.log 2>&1 || { echo "CONVERT_FAILED seed \$s"; exit 1; }
  grep -a "Final Summary" -A8 \$T/logs/evalpy_seed\$s.log > \$T/results/evalpy_final_summary_seed\$s.txt
  gcloud storage rsync -r --quiet \$T/results $G/tests/$TAG/results >/dev/null 2>&1
  echo "SEED_DONE \$s"
done
/root/hpsv3-venv/bin/python eval_hpsv3.py --save_dir \$T/images --result_dir \$T/results > \$T/logs/hpsv3_$POOL.log 2>&1 || { echo HPSV3_FAILED; exit 1; }
gcloud storage rsync -r --quiet \$T/results $G/tests/$TAG/results >/dev/null 2>&1
gcloud storage cp --quiet \$T/logs/* $G/tests/$TAG/logs/ >/dev/null 2>&1
echo TEST_DONE
EOF

rsync -a --exclude .git --exclude __pycache__ --exclude output.log -e "ssh $KEY" ~/npo/ ubuntu@$IP:npo_src/ \
  && scp -q $KEY "$OUT/node_script_$POOL.sh" "$DIR/orig_eval_check/run_orig_eval.py" "$DIR/orig_eval_check/convert_evalpy_csv.py" ubuntu@$IP: \
  && ssh $KEY ubuntu@$IP "sudo rm -rf /root/npo && sudo cp -r ~/npo_src /root/npo && sudo cp ~/node_script_$POOL.sh ~/run_orig_eval.py ~/convert_evalpy_csv.py /root/ && sudo bash -c 'nohup bash /root/node_script_$POOL.sh > /root/test.out 2>&1 &'" \
  || { log "TEST_FAILED: could not start on the node"; exit 1; }
log "test $TAG started (seeds $SEEDS)"
while true; do
  sleep 60
  st=$(ssh $KEY ubuntu@$IP "sudo cat /root/test.out 2>/dev/null | grep -o 'TEST_DONE\|[A-Z0-9_]*FAILED[ a-z0-9]*' | tail -1" 2>/dev/null)
  if [ -n "$st" ]; then break; fi
  if ! gcloud compute instances describe "$NODE" --zone "$ZONE" --format="value(status)" 2>/dev/null | grep -q RUNNING; then
    st="NODE_LOST (spot preemption); rerun these seeds"; break
  fi
  prog=$(ssh $KEY ubuntu@$IP "sudo bash -c 'grep -a SEED_DONE /root/test.out; ls -t /root/tests/$TAG/logs/evalpy_seed*.log 2>/dev/null | head -1 | xargs -r grep -a ^Samples | tail -1; grep -ac HPSv3 /root/tests/$TAG/logs/hpsv3_$POOL.log 2>/dev/null'" 2>/dev/null | tr '\n' ' ')
  log "running: $prog"
done
log "node finished with: $st"
ssh $KEY ubuntu@$IP "sudo bash -c 'cat /root/test.out; for f in /root/tests/$TAG/logs/*.log; do echo ---- \$f; tail -c 1500 \$f; done; echo ---- setup; tail -5 /root/setup_main.log /root/setup_hpsv3.log'" > "$OUT/log_tails_$POOL.txt" 2>/dev/null
if [ "$st" = TEST_DONE ]; then log "TEST_OK seeds $SEEDS"; else log "TEST_FAILED: $st"; fi
