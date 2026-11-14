#!/usr/bin/env python
"""Run the user's ORIGINAL eval.py unchanged (via runpy), preparing only the environment around it:

  - utils/ package: the repo upload has the metric wrappers at the top level while eval.py imports
    utils.*, so the package of symlinks is assembled first (train_nash_sana.ensure_utils_package);
  - `turtle` stub when tkinter is missing (open_clip has a stray `from turtle import forward`);
  - datasets.load_dataset is replaced by a function that returns the 500 captions of pickapic_v1
    test_unique, so the token written in eval.py is never sent anywhere and nothing but that split
    is downloaded.

Every line of eval.py itself (pipelines, generation, metric calls, averages, win rates, CSV) runs as is.
Usage (cwd is set to the repo root):  python run_orig_eval.py <eval.py arguments>
"""
import os
import runpy
import sys
import types

REPO = "/root/npo"
os.chdir(REPO)
sys.path.insert(0, REPO)

import train_nash_sana as t  # noqa: E402

t.ensure_utils_package()
try:
    import tkinter  # noqa: F401
except ImportError:
    stub = types.ModuleType("turtle")
    stub.forward = None
    sys.modules.setdefault("turtle", stub)

from eval_sana import load_test_prompts  # noqa: E402

PROMPTS = load_test_prompts()
print(f"[run_orig_eval] {len(PROMPTS)} test_unique captions", flush=True)

import datasets  # noqa: E402


def _load_dataset(name, *a, **k):
    assert name == "pickapic-anonymous/pickapic_v1" and k.get("split") == "test_unique", (name, k.get("split"))
    return [{"caption": c} for c in PROMPTS]


datasets.load_dataset = _load_dataset
sys.argv = ["eval.py"] + sys.argv[1:]
runpy.run_path(os.path.join(REPO, "eval.py"), run_name="__main__")
