"""Run train_nash_sana.py on N spot A100 workers of the Ray cluster, surviving preemption.

Submit from the Ray head (see launchers/sana_submit.sh):

    ray job submit --working-dir . -- python3 launchers/sana_ray_job.py --num_gpus 4 <train args>

Each GPU worker runs one training process (rank); gradients are averaged across ranks once per optimizer step, so N GPUs give
N times more pairs per optimizer step. If any worker is preempted, the whole group is torn down,
new workers are requested, and training resumes from the last checkpoint in --gcs_dir.
"""
import ctypes
import os
import shlex
import signal
import subprocess
import sys
import time

import ray

MAX_RESTARTS = 100


@ray.remote(num_gpus=1, max_restarts=0)
class Rank:
    def ip(self):
        return ray.util.get_node_ip_address()

    def ping(self):
        return True

    def run(self, argv, rank, world_size, master_addr, master_port):
        import socket
        print(f"[sana_ray_job] rank {rank}/{world_size} host={socket.gethostname()} "
              f"ip={ray.util.get_node_ip_address()} master={master_addr}:{master_port}", flush=True)
        cmd = (
            "set -eo pipefail; "
            "bash launchers/sana_setup_env.sh; "
            "source ${VENV:-$HOME/sana-venv}/bin/activate; "
            "exec python train_nash_sana.py " + " ".join(shlex.quote(a) for a in argv)
        )
        env = dict(os.environ, WORLD_SIZE=str(world_size), RANK=str(rank), MASTER_ADDR=master_addr,
                   MASTER_PORT=str(master_port), TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD="1",
                   TOKENIZERS_PARALLELISM="false", PYTHONUNBUFFERED="1")

        def die_with_parent():  # if this actor is killed, the training process must die too
            ctypes.CDLL("libc.so.6").prctl(1, signal.SIGKILL)

        return subprocess.run(["bash", "-c", cmd], cwd=os.getcwd(), env=env, preexec_fn=die_with_parent).returncode


def alive(actor, timeout=20):
    try:
        ray.get(actor.ping.remote(), timeout=timeout)
        return True
    except ray.exceptions.GetTimeoutError:
        return True   # busy is not dead (ping queues behind run())
    except Exception:
        return False


def main():
    argv = sys.argv[1:]
    num_gpus = 1
    if "--num_gpus" in argv:
        i = argv.index("--num_gpus")
        num_gpus = int(argv[i + 1])
        del argv[i:i + 2]
    # --node_resource NAME: run on nodes that advertise the custom Ray resource NAME (one unit per
    # GPU) instead of asking Ray for a generic GPU. Used for dedicated nodes that other jobs on the
    # cluster must not take (they are started with --num-gpus=0 --resources='{"NAME": 1}').
    node_resource = None
    if "--node_resource" in argv:
        i = argv.index("--node_resource")
        node_resource = argv[i + 1]
        del argv[i:i + 2]
    rank_cls = Rank.options(num_gpus=0, resources={node_resource: 1}) if node_resource else Rank
    if not any(a.startswith("--gcs_dir") for a in argv):
        print("[sana_ray_job] WARNING: no --gcs_dir given; a preempted run cannot resume.", flush=True)

    ray.init(address="auto")
    for attempt in range(MAX_RESTARTS + 1):
        actors = [rank_cls.remote() for _ in range(num_gpus)]
        try:
            ips = ray.get([a.ip.remote() for a in actors])   # waits for the autoscaler
            print(f"[sana_ray_job] attempt {attempt}: workers {ips}", flush=True)
            port = 29500 + attempt % 200
            pending = [a.run.remote(argv, r, num_gpus, ips[0], port) for r, a in enumerate(actors)]
            failed_rc = None
            while pending and failed_rc is None:
                done, pending = ray.wait(pending, num_returns=1)
                rc = ray.get(done[0])            # raises if that worker died
                if rc != 0:
                    failed_rc = rc
            if failed_rc is None:
                print("[sana_ray_job] training finished", flush=True)
                return 0
            # A rank exited with an error. If a worker was lost, the others crash too -> restart.
            time.sleep(90)   # give Ray time to notice a dead node
            if all(alive(a) for a in actors):
                print(f"[sana_ray_job] training failed with exit code {failed_rc} (no worker lost); giving up",
                      flush=True)
                return failed_rc
            print("[sana_ray_job] a worker was lost; restarting the group", flush=True)
        except ray.exceptions.RayError as e:
            print(f"[sana_ray_job] worker lost ({type(e).__name__}); restarting the group", flush=True)
        finally:
            for a in actors:
                ray.kill(a)
        time.sleep(20)
    print("[sana_ray_job] too many restarts; giving up", flush=True)
    return 1


if __name__ == "__main__":
    sys.exit(main())
