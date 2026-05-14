"""
LongBench evaluation for Llama-2-7B-chat (4K context).
Use Case: Sole Decode Optimization — evaluating delta-compressed KV cache
against H2O / KVMerger baselines on the 9-task LongBench subset.

Model:  meta-llama/Llama-2-7b-chat-hf  (native 4K window; inputs are
        mid-truncated to 4K by the lm-eval LongBench tasks, matching the
        THUDM evaluation protocol used by the baseline papers)
Thresholds tested: row_thresh = 15 (~50% KV compression) and 17
Tasks:  gov_report, multinews, narrativeqa, qasper, multifieldqa_en,
        trec, 2wikimqa, triviaqa, passage_retrieval_en
"""

import os
os.environ["HF_DATASETS_TRUST_REMOTE_CODE"] = "1"
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
import sys
import json
import argparse
import torch
import numpy as np
import re
import gc
import time
import subprocess
from datetime import datetime

import globVR
import glob_set
import lm_eval.api.registry
from lm_eval.evaluator import simple_evaluate
from transformers import AutoConfig, AutoModelForCausalLM

from modeling_llama import LlamaForCausalLM, LlamaConfig
from deltaDecoding import HybridCompressedCache

# --- fix lm_eval JSON crash on torch types ---
_original_json_default = json.JSONEncoder.default

def safe_json_default(self, obj):
    if isinstance(obj, (torch.dtype, torch.device)):
        return str(obj)
    return _original_json_default(self, obj)

json.JSONEncoder.default = safe_json_default
# ---------------------------------------------

AutoConfig.register("llama", LlamaConfig, exist_ok=True)
AutoModelForCausalLM.register(LlamaConfig, LlamaForCausalLM, exist_ok=True)

LIMIT = None
MODEL_ID = "meta-llama/Llama-2-7b-chat-hf"
MAX_LENGTH = 4096  # Llama-2-7B native context window

# 9-task LongBench subset matching the KVMerger / H2O comparison table
LONGBENCH_TASKS = [
    "longbench_gov_report",
    "longbench_multi_news",
    "longbench_narrativeqa",
    "longbench_qasper",
    "longbench_multifieldqa_en",
    "longbench_trec",
    "longbench_2wikimqa",
    "longbench_triviaqa",
    "longbench_passage_retrieval_en",
]

# ==========================================
# EXPERIMENT DEFINITIONS
# ==========================================
EXPERIMENT_DEFINITIONS = {
    "baseline": {
        "injector": lambda args: {
            "delta_decode": False,
            "delta_pf_key_on": 0,
            "delta_mlp": "Regular",
            "flash": True,
        }
    },
    # Regular prefill + delta-compressed KV cache during decoding only.
    # This is the "Sole Decode Optimization" use case from the eval plan.
    "decode_only_delta": {
        "grid": {
            "row_thresh": [13,10],
            "window_size": [100],
            "row_sim": ["euclidean"],
        },
        "arg_builder": lambda p: [
            "--row_thresh",  str(p["row_thresh"]),
            "--window_size", str(p["window_size"]),
            "--row_sim",     str(p["row_sim"]),
        ],
        "injector": lambda args: {
            "delta_decode": True,
            "window_size": args.window_size,
            "row_delta_threshold": args.row_thresh,
            "row_similarity_metric": args.row_sim,
            "delta_pf_key_on": 0,
            "flash": True,
            "delta_type": "row",
            "divide_to": 0,
            "chunk_size": 512,
        }
    },
}

# ==========================================
# HELPERS
# ==========================================
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, torch.Tensor):
            return obj.item() if obj.numel() == 1 else obj.tolist()
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def get_next_id(directory, prefix, suffix=""):
    if not os.path.exists(directory):
        return 1
    existing_ids = set()
    pattern = re.compile(rf"^{re.escape(prefix)}(\d+){re.escape(suffix)}$")
    for item in os.listdir(directory):
        m = pattern.match(item)
        if m:
            existing_ids.add(int(m.group(1)))
    next_id = 1
    while next_id in existing_ids:
        next_id += 1
    return next_id


def get_or_create_config_dir(base_exp_dir, delta_config, force_dir_name=None):
    os.makedirs(base_exp_dir, exist_ok=True)

    if force_dir_name:
        dir_path = os.path.join(base_exp_dir, force_dir_name)
        os.makedirs(os.path.join(dir_path, "tasks"), exist_ok=True)
        with open(os.path.join(dir_path, "config_params.json"), 'w') as f:
            json.dump(delta_config, f, indent=4, cls=NpEncoder)
        print(f"[Config] Using forced directory: {dir_path}", flush=True)
        return dir_path

    for item in sorted(os.listdir(base_exp_dir)):
        dir_path = os.path.join(base_exp_dir, item)
        if os.path.isdir(dir_path) and item.startswith("config_"):
            param_file = os.path.join(dir_path, "config_params.json")
            if os.path.exists(param_file):
                try:
                    with open(param_file, 'r') as f:
                        loaded = json.load(f)
                    if all(k in loaded and loaded[k] == v for k, v in delta_config.items()):
                        print(f"[Config] Found existing directory: {dir_path}", flush=True)
                        return dir_path
                except Exception:
                    pass

    next_id = get_next_id(base_exp_dir, prefix="config_")
    new_dir = os.path.join(base_exp_dir, f"config_{next_id}")
    os.makedirs(os.path.join(new_dir, "tasks"), exist_ok=True)
    with open(os.path.join(new_dir, "config_params.json"), 'w') as f:
        json.dump(delta_config, f, indent=4, cls=NpEncoder)
    print(f"[Config] Created new directory: {new_dir}", flush=True)
    return new_dir


def save_single_task_result(config_dir, task_name, result_data):
    tasks_dir = os.path.join(config_dir, "tasks")
    file_path  = os.path.join(tasks_dir, f"{task_name}.json")
    lock_path  = file_path + ".lock"

    while os.path.exists(lock_path):
        time.sleep(0.2)
    try:
        open(lock_path, 'w').close()
        with open(file_path, 'w') as f:
            json.dump(result_data, f, indent=4, cls=NpEncoder)
        print(f"[IO] Saved result to {os.path.basename(file_path)}", flush=True)
    finally:
        if os.path.exists(lock_path):
            os.remove(lock_path)


def run_single_pass(lm_model, task, eval_config, glob_settings):
    setattr(globVR, 'time_internal', True)

    for attr in ('spars', 'mlp_spars', 'mlp_spars_count'):
        if hasattr(globVR, attr):
            setattr(globVR, attr, 0.0 if attr != 'mlp_spars_count' else 0)
    for attr in ('latency_stats', 'latency_events', 'sequence_lengths'):
        if hasattr(globVR, attr):
            setattr(globVR, attr, {} if attr == 'latency_stats' else [])
    globVR.kv_compression_samples = []

    torch.cuda.empty_cache()
    gc.collect()

    start = time.time()
    try:
        eval_output = simple_evaluate(
            model=lm_model,
            tasks=[task],
            num_fewshot=eval_config["shot"],
            limit=eval_config["limit"],
            apply_chat_template=True,
        )
    except Exception as e:
        print(f"    - [Error] simple_evaluate failed for {task}: {e}", flush=True)
        return None, 0
    total_time = time.time() - start
    print(f"    - Task '{task}' completed in {total_time:.2f}s.", flush=True)

    glob_set.resolve_latency_events()

    latency_breakdown = {}
    if hasattr(globVR, 'latency_stats'):
        for metric, stats in globVR.latency_stats.items():
            if stats['calls'] > 0:
                latency_breakdown[metric] = stats['time_ms'] / stats['calls']

    total_avg_ms = latency_breakdown.get(
        'time_prefill_forward_total',
        latency_breakdown.get('time_forward_total', 0.0)
    )

    seq_lengths = getattr(globVR, 'sequence_lengths', [])
    benchmark_stats = {
        "num_passages": len(seq_lengths),
        "seq_len_avg": float(np.mean(seq_lengths)) if seq_lengths else 0.0,
        "seq_len_min": int(np.min(seq_lengths))    if seq_lengths else 0,
        "seq_len_max": int(np.max(seq_lengths))    if seq_lengths else 0,
        "seq_len_std": float(np.std(seq_lengths))  if seq_lengths else 0.0,
    }

    kv_samples = getattr(globVR, 'kv_compression_samples', [])
    avg_kv_compression = float(np.mean(kv_samples)) if kv_samples else 0.0

    raw_metrics = eval_output["results"].get(task, {})

    # LongBench metrics vary by task — pick the first non-zero value
    primary_acc = (
        raw_metrics.get("qa_f1_score,none") or
        raw_metrics.get("summary_rouge_l,none") or
        raw_metrics.get("rouge_score,none") or
        raw_metrics.get("retrieval_score,none") or
        raw_metrics.get("rougeL,none") or
        raw_metrics.get("f1,none") or
        raw_metrics.get("acc,none") or
        raw_metrics.get("acc_norm,none") or
        raw_metrics.get("exact_match,none") or
        0.0
    )

    result_data = {
        "task_name": task,
        "timestamp": datetime.now().isoformat(),
        "shot": eval_config["shot"],
        "sparsity": getattr(globVR, 'spars', 0.0),
        "mlp_spars": getattr(globVR, 'mlp_spars', 0.0),
        "avg_kv_compression_pct": avg_kv_compression,
        "benchmark_stats": benchmark_stats,
        "accuracy": primary_acc,
        "timings": {
            "total_avg_ms": total_avg_ms,
            "latency_breakdown_ms": latency_breakdown,
        },
        "metrics": raw_metrics,
    }
    return result_data, total_time


def run_worker_process(args, experiment_dir):
    torch.set_grad_enabled(False)

    exp_def      = EXPERIMENT_DEFINITIONS[args.experiment_type]
    glob_settings = exp_def["injector"](args)

    print(f"[Worker] Experiment settings: {glob_settings}", flush=True)

    model_args = (
        f"pretrained={MODEL_ID},trust_remote_code=True,"
        f"dtype=bfloat16,attn_implementation=eager,max_length={MAX_LENGTH}"
    )
    eval_config = {"shot": args.shot, "limit": LIMIT, "batch_size": 1}

    # Safe defaults before model load
    setattr(globVR, 'delta_pf_key_on', 0)
    setattr(globVR, 'delta_mlp',      'Regular')
    setattr(globVR, 'delta_decode',   False)

    print(f"[Worker] Loading model: {MODEL_ID}...", flush=True)
    try:
        model_class = lm_eval.api.registry.get_model("hf")
        lm_model = model_class.create_from_arg_string(
            model_args,
            {"batch_size": eval_config["batch_size"], "device": "cuda"},
        )
    except Exception as e:
        print(f"[Fatal Worker Error] Model load failed: {e}", flush=True)
        return

    # Inject experiment globals after model load
    print(f"[Worker] Applying {args.experiment_type} settings...", flush=True)
    for key, value in glob_settings.items():
        setattr(globVR, key, value)

    # Monkey-patch generate() to inject HybridCompressedCache when decoding
    if getattr(globVR, 'delta_decode', False):
        print("[Worker] Patching model.generate() for HybridCompressedCache...", flush=True)
        original_generate = lm_model.model.generate

        def generate_with_custom_cache(inputs=None, *gen_args, **gen_kwargs):
            src = inputs if inputs is not None else gen_kwargs.get('input_ids')
            bsz = src.shape[0] if src is not None else 1
            gen_kwargs['past_key_values'] = HybridCompressedCache(
                config=lm_model.model.config,
                batch_size=bsz,
                dtype=lm_model.model.dtype,
                exact_window_size=getattr(globVR, 'window_size', 100),
            )
            gen_kwargs['use_cache'] = True
            return original_generate(inputs, *gen_args, **gen_kwargs)

        lm_model.model.generate = generate_with_custom_cache

    force_name = "config_baseline" if args.experiment_type == "baseline" else None
    config_dir  = get_or_create_config_dir(experiment_dir, glob_settings, force_dir_name=force_name)

    for task in LONGBENCH_TASKS:
        print(f"--- Running Task: {task} ---", flush=True)
        results, total_time = run_single_pass(lm_model, task, eval_config, glob_settings)

        if results:
            results['total_benchmark_time_s'] = total_time
            results["parameters"]             = glob_settings
            results["experiment_type"]        = args.experiment_type
            results["model_id"]               = MODEL_ID

            print(
                f"Task: {task} | Acc: {results.get('accuracy', 0.0):.4f} | "
                f"Time: {total_time:.2f}s",
                flush=True,
            )
            print(json.dumps(results, indent=4, cls=NpEncoder), flush=True)
            save_single_task_result(config_dir, task, results)
        else:
            print(f"[Error] No results for task {task}.", flush=True)

        gc.collect()
        torch.cuda.empty_cache()

    print("[Worker] Finished.", flush=True)


# ==========================================
# MAIN ENTRY POINT
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "LongBench evaluation — Llama-2-7B-chat, 4K context. "
            "Sole Decode Optimization: delta-compressed KV cache (row_thresh=15/17)."
        )
    )
    parser.add_argument('--mode', type=str, default='master', choices=['master', 'worker'])
    parser.add_argument(
        '--experiment_type', type=str, required=True,
        choices=list(EXPERIMENT_DEFINITIONS.keys()),
    )
    parser.add_argument('--shot',     default=0, type=int)
    parser.add_argument('--exp_name', default=None, type=str,
                        help="Custom folder name for experiment output")

    # Decode experiment args
    parser.add_argument('--row_thresh',  default=15,           type=float,
                        help="Row-delta threshold. 15 ≈ 50%% KV compression.")
    parser.add_argument('--window_size', default=100,          type=int,
                        help="Exact (uncompressed) token window size for HybridCompressedCache.")
    parser.add_argument('--row_sim',     default="euclidean",  type=str)

    args = parser.parse_args()

    base_storage_path = os.path.join("experiments", "longbench")
    os.makedirs(base_storage_path, exist_ok=True)

    if args.mode == 'master':
        if args.exp_name is None:
            auto_id = get_next_id(base_storage_path, prefix=f"experiment_{args.experiment_type}_")
            args.exp_name = f"experiment_{args.experiment_type}_{auto_id}"

        exp_def    = EXPERIMENT_DEFINITIONS[args.experiment_type]
        grid_params = exp_def.get("grid", {})

        import itertools
        keys      = list(grid_params.keys())   if grid_params else []
        values    = list(grid_params.values()) if grid_params else []
        param_grid = list(itertools.product(*values)) if grid_params else [{}]

        print(f"[Master] Experiment:  {args.experiment_type}",  flush=True)
        print(f"[Master] Grid params: {keys}",                  flush=True)
        print(f"[Master] Combos:      {len(param_grid)}",       flush=True)
        print(f"[Master] Output:      {args.exp_name}",         flush=True)
        print(f"[Master] Tasks:       {LONGBENCH_TASKS}",       flush=True)

        for i, combo in enumerate(param_grid):
            current_params = dict(zip(keys, combo)) if keys else {}
            print(f"\n=== Step {i+1}/{len(param_grid)}: {current_params} ===", flush=True)

            cmd = [
                sys.executable, sys.argv[0],
                "--mode",            "worker",
                "--experiment_type", args.experiment_type,
                "--exp_name",        args.exp_name,
                "--shot",            str(args.shot),
            ]
            if "arg_builder" in exp_def:
                cmd.extend(exp_def["arg_builder"](current_params))

            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError:
                print(f"[Master] Worker failed for {current_params}. Continuing...", flush=True)

    elif args.mode == 'worker':
        if args.exp_name is None:
            args.exp_name = "default_worker_run"
        experiment_dir = os.path.join(base_storage_path, args.exp_name)
        run_worker_process(args, experiment_dir)
