#!/bin/bash
# usage: make_node.sh <name> <pool>   -> tries zones until a spot A100 80GB is obtained
# A pool is the set of nodes of one run. If the first-choice zone has no capacity, every other zone
# that offers the machine type is tried, so no single zone can block us.
NAME="$1"; POOL="$2"
DIR="$(cd "$(dirname "$0")" && pwd)"
shuffle() { tr ' ' '\n' | shuf | tr '\n' ' '; }
US="$(echo us-central1-a us-central1-c us-east5-a us-east5-b | shuffle) us-east4-c"
EU="$(echo europe-west4-a europe-west4-b | shuffle)"
ASIA="asia-southeast1-c"
# Observed 2026-11-11: US spot A100-80GB nodes were preempted within 7-15 minutes (3 of 4), while
# asia-southeast1-c stayed up, so both pools try Asia first and fall back to Europe, then the US.
ZONES="${ZONES:-$ASIA $EU $US}"
for Z in $ZONES; do
  if gcloud compute instances create "$NAME" --zone "$Z" --machine-type a2-ultragpu-1g \
      --provisioning-model SPOT --instance-termination-action DELETE --maintenance-policy TERMINATE \
      --max-run-duration 96h \
      --image-family common-cu129-ubuntu-2204-nvidia-580 --image-project deeplearning-platform-release \
      --boot-disk-size "${DISK_GB:-150}GB" --boot-disk-type pd-balanced \
      --service-account prism-training-sa@unity-prism-dev.iam.gserviceaccount.com --scopes cloud-platform \
      --metadata npo-pool="$POOL",enable-oslogin=FALSE,ssh-keys="ubuntu:$(cat "$DIR/worker_key.pub") jiaming.hu@jiaming-ray-head" \
      --metadata-from-file startup-script="$DIR/startup.sh" \
      --labels owner=jiaming,project=npo-sana,pool="$POOL" \
      --format="value(name,zone.basename(),networkInterfaces[0].networkIP)" > "/tmp/npo_create_${NAME}.out" 2> "/tmp/npo_create_${NAME}_$Z.err"; then
    echo "CREATED $NAME pool=$POOL zone=$Z"
    exit 0
  fi
done
echo "FAILED $NAME pool=$POOL: no zone had capacity"
exit 1
