#!/bin/bash
# Run the user's test (eval.py protocol) on a temporary spot A100, then delete the node.
#
#   ./run_test.sh <tag> "<name>|<label>|<checkpoint>" [more specs ...]
#
#   MODEL_TYPE=sana (default): <checkpoint> is a gs:// path of a 'transformer' folder.
#   MODEL_TYPE=sd15          : <checkpoint> is a Hugging Face repo (or gs:// folder) that contains 'unet'.
#   TEST_POOL (default t)    : node pool label; use another letter to run two tests at the same time.
#   KEEP_NODE=1              : leave the node running (default: delete it when the test ends).
#
# Protocol (user, 2026-11-12): the existing eval.py test unchanged — 500 prompts of pickapic_v1
# test_unique as they are, metrics PickScore / HPS (HPSv2) / CLIP / Aesthetics / ImageReward, averages and
# win rate against the base model — with 5 seeds (0-4) and one added column, HPSv3.
# Results: gs://prism-training-apac/jiaming/npo-sana/tests/<tag>/ and ~/npo_sana_dev/tests/<tag>/.
# The base model's images and scores are cached in the bucket so they are computed once per model type.
set -uo pipefail
TAG="$1"; shift
DIR="$(cd "$(dirname "$0")" && pwd)"
G=gs://prism-training-apac/jiaming/npo-sana
POOL="${TEST_POOL:-t}"
MT="${MODEL_TYPE:-sana}"
if [ "$MT" = sd15 ]; then BASE_NAME="Orig. SD1.5"; BASE_DIR="Orig._SD1.5"; CACHE="$G/tests/_baseline_cache_sd15"; CK_FLAG="--model_type sd15 --unet_checkpoints"
else BASE_NAME="Orig. Sana"; BASE_DIR="Orig._Sana"; CACHE="$G/tests/_baseline_cache"; CK_FLAG="--transformer_checkpoints"; fi
KEY="-i $HOME/.ssh/ray_worker_key -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -o ConnectTimeout=20"
OUT="$DIR/tests/$TAG"; mkdir -p "$OUT"
log() { echo "$(date '+%m/%d %H:%M:%S') $*"; }

# ---- 1. node ----
NODE=$(gcloud compute instances list --filter="labels.project=npo-sana AND labels.pool=$POOL AND status=RUNNING" --format="value(name)" | head -1)
if [ -z "$NODE" ]; then
  NODE="npo-test-$POOL-$(date +%d%H%M%S)"
  log "creating test node $NODE"
  "$DIR/nodes/make_node.sh" "$NODE" "$POOL" || { log "TEST_FAILED: could not get a GPU"; exit 1; }
fi
ZONE=$(gcloud compute instances list --filter="name=$NODE" --format="value(zone.basename())")
IP=$(gcloud compute instances list --filter="name=$NODE" --format="value(networkInterfaces[0].networkIP)")
log "test node $NODE ($ZONE, $IP)"
cleanup() {
  if [ "${KEEP_NODE:-0}" != 1 ]; then
    log "deleting test node $NODE"
    gcloud compute instances delete "$NODE" --zone "$ZONE" --quiet >/dev/null 2>&1
  fi
}
trap cleanup EXIT
for i in $(seq 1 60); do ssh $KEY ubuntu@$IP 'echo up' 2>/dev/null | grep -q up && break; sleep 10; done
# a fresh node is still installing pip / ray in its boot script; setting up before that finishes fails
for i in $(seq 1 90); do ssh $KEY ubuntu@$IP 'sudo grep -q NPO_NODE_READY /var/log/npo-startup.log' 2>/dev/null && break; sleep 10; done
log "node boot script finished"

# ---- 2. node-side script ----
CKPT_ARGS=""; EPOCH_ARGS=""; NAME_ARGS="\"$BASE_NAME\""; DL=""
n=0
for spec in "$@"; do
  IFS='|' read -r name epoch path <<< "$spec"
  n=$((n+1))
  if [[ "$path" == gs://* ]]; then
    DL="$DL
mkdir -p /root/test_ckpts/$TAG/m$n && gcloud storage cp -r --quiet '${path%/}' /root/test_ckpts/$TAG/m$n/ >/dev/null 2>&1 || { echo DOWNLOAD_FAILED m$n; exit 1; }"
    CKPT_ARGS="$CKPT_ARGS /root/test_ckpts/$TAG/m$n"
  else
    CKPT_ARGS="$CKPT_ARGS $path"
  fi
  EPOCH_ARGS="$EPOCH_ARGS $epoch"
  NAME_ARGS="$NAME_ARGS \"$name\""
done
cat > "$OUT/node_script.sh" <<EOF
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
T=/root/tests/$TAG; mkdir -p \$T/images \$T/results
gcloud storage rsync -r --quiet $CACHE/images \$T/images >/dev/null 2>&1
gcloud storage rsync -r --quiet $CACHE/results \$T/results >/dev/null 2>&1
$DL
/root/sana-venv/bin/python eval_sana.py $CK_FLAG $CKPT_ARGS --epochs $EPOCH_ARGS --names $NAME_ARGS \\
  --seeds 0 1 2 3 4 --batch_size 32 --save_dir \$T/images --result_dir \$T/results > \$T/gen.log 2>&1 || { echo GEN_FAILED; exit 1; }
/root/hpsv3-venv/bin/python eval_hpsv3.py --save_dir \$T/images --result_dir \$T/results > \$T/hpsv3.log 2>&1 || { echo HPSV3_FAILED; exit 1; }
/root/sana-venv/bin/python eval_summarize.py --result_dir \$T/results > \$T/summary.log 2>&1 || { echo SUMMARY_FAILED; exit 1; }
gcloud storage rsync -r --quiet \$T/results $G/tests/$TAG/results >/dev/null 2>&1
gcloud storage cp --quiet \$T/gen.log \$T/hpsv3.log \$T/summary.log $G/tests/$TAG/ >/dev/null 2>&1
gcloud storage rsync -r --quiet "\$T/images/$BASE_DIR" "$CACHE/images/$BASE_DIR" >/dev/null 2>&1
gcloud storage rsync -r --quiet "\$T/results/$BASE_DIR" "$CACHE/results/$BASE_DIR" >/dev/null 2>&1
echo TEST_DONE
EOF

# ---- 3. run ----
rsync -a --exclude .git --exclude __pycache__ --exclude output.log -e "ssh $KEY" ~/npo/ ubuntu@$IP:npo_src/ \
  && scp -q $KEY "$OUT/node_script.sh" ubuntu@$IP:node_script_$TAG.sh \
  && ssh $KEY ubuntu@$IP "sudo rm -rf /root/npo && sudo cp -r ~/npo_src /root/npo && sudo cp ~/node_script_$TAG.sh /root/node_script_$TAG.sh && sudo bash -c 'nohup bash /root/node_script_$TAG.sh > /root/test_$TAG.out 2>&1 &'" \
  || { log "TEST_FAILED: could not start on the node"; exit 1; }
log "test $TAG ($MT) started with $n checkpoint(s)"
while true; do
  sleep 60
  st=$(ssh $KEY ubuntu@$IP "sudo cat /root/test_$TAG.out 2>/dev/null | grep -o 'TEST_DONE\|[A-Z0-9_]*FAILED[ a-z0-9]*' | tail -1" 2>/dev/null)
  if [ -n "$st" ]; then break; fi
  if ! gcloud compute instances describe "$NODE" --zone "$ZONE" --format="value(status)" 2>/dev/null | grep -q RUNNING; then
    st="NODE_LOST (spot preemption); rerun the test"; break
  fi
  prog=$(ssh $KEY ubuntu@$IP "sudo bash -c 'grep -a \"prompts\$\" /root/tests/$TAG/gen.log 2>/dev/null | tail -1; grep -ac \"mean HPSv3\" /root/tests/$TAG/hpsv3.log 2>/dev/null'" 2>/dev/null | tr '\n' ' ')
  log "running: $prog"
done
log "node finished with: $st"
ssh $KEY ubuntu@$IP "sudo bash -c 'cat /root/tests/$TAG/summary.log 2>/dev/null; echo; echo ---- errors; grep -a \"Error\|Traceback\" -A3 /root/tests/$TAG/gen.log /root/tests/$TAG/hpsv3.log 2>/dev/null | tail -20'" > "$OUT/summary_and_errors.txt" 2>/dev/null
gcloud storage cp --quiet "$G/tests/$TAG/results/summary.md" "$G/tests/$TAG/results/summary.json" "$OUT/" >/dev/null 2>&1
if [ "$st" = TEST_DONE ]; then log "TEST_OK results in $OUT"; else log "TEST_FAILED: $st"; fi
