#!/bin/bash
# Boot script for a manually created spot A100 node: join the jiaming-ray cluster with a
# private resource ("npo_<pool>") so that only the NPO Sana run of that pool is scheduled here.
exec > /var/log/npo-startup.log 2>&1
set -x
export DEBIAN_FRONTEND=noninteractive HOME=/root
POOL=$(curl -s -H "Metadata-Flavor: Google" http://metadata.google.internal/computeMetadata/v1/instance/attributes/npo-pool)
for i in $(seq 1 60); do nvidia-smi && break; sleep 10; done
python3 -m pip --version || { apt-get -qq update; apt-get -qq install -y python3-pip; }
python3 -m pip install -q "ray[default]==2.56.1" google-api-python-client virtualenv
ray stop || true
ray start --address=10.150.0.3:6379 --num-cpus=12 --num-gpus=0 --resources="{\"npo_${POOL}\": 1}"
echo NPO_NODE_READY
