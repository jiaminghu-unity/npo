#!/bin/bash
# Test the Online-DPO control at one step as soon as its milestone is archived:
#   ./auto_test_odpo.sh <step>
# Pins the checkpoint (the 4-newest retention would delete it) and runs the standard test
# (500 prompts x 5 seeds, six metrics). Log: tests_odpo<step>.log, results: tests/odpo-step<step>/.
cd "$(dirname "$0")"
STEP="$1"
S6=$(printf '%06d' "$STEP")
G=gs://prism-training-apac/jiaming/npo-sana
log() { echo "$(date '+%m/%d %H:%M:%S') $*"; }
until gcloud storage ls "$G/sana-onlinedpo-avgrank-v1/milestones/step-$S6/transformer/diffusion_pytorch_model.safetensors" >/dev/null 2>&1; do sleep 120; done
gcloud storage cp -r --quiet "$G/sana-onlinedpo-avgrank-v1/milestones/step-$S6/transformer" "$G/tests/_pinned/onlinedpo-step$STEP/" >/dev/null 2>&1
log "pinned onlinedpo step $STEP: $(gcloud storage ls -l "$G/tests/_pinned/onlinedpo-step$STEP/transformer/" | grep -c safetensors) weight file"
./run_test.sh "odpo-step$STEP" "Online-DPO-avgrank|$STEP|$G/tests/_pinned/onlinedpo-step$STEP/transformer" > "tests_odpo$STEP.log" 2>&1
log "run_test.sh exited: $(grep -a 'TEST_OK\|TEST_FAILED' "tests_odpo$STEP.log" | tail -1)"
