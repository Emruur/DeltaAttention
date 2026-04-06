import os
os.environ["HF_DATASETS_TRUST_REMOTE_CODE"] = "1" # ADD THIS LINE
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

# ==========================================
# IMPORTS & REGISTRATION
# ==========================================

import globVR
import glob_set
import lm_eval.api.registry
from lm_eval.evaluator import simple_evaluate
from lm_eval.tasks import TaskManager, get_task_dict # <-- ADD THIS
from transformers import AutoConfig, AutoModelForCausalLM

from modeling_bitnet import BitNetForCausalLM, BitNetConfig
from modeling_llama import LlamaForCausalLM, LlamaConfig





# --- ADD THIS GLOBALLY TO FIX THE LM_EVAL JSON CRASH ---
_original_json_default = json.JSONEncoder.default

def safe_json_default(self, obj):
    if isinstance(obj, (torch.dtype, torch.device)):
        return str(obj)
    return _original_json_default(self, obj)

json.JSONEncoder.default = safe_json_default
# --------------------------

# Register BitNet
AutoConfig.register("bitnet", BitNetConfig, exist_ok=True)
AutoModelForCausalLM.register(BitNetConfig, BitNetForCausalLM, exist_ok=True)

# Register LLaMA
AutoConfig.register("llama", LlamaConfig, exist_ok=True)
AutoModelForCausalLM.register(LlamaConfig, LlamaForCausalLM, exist_ok=True)


# ==========================================
# EXPERIMENT DEFINITIONS
# ==========================================
EXPERIMENT_DEFINITIONS = {
    "scale_delta": {
        "grid": {
            "scale": [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0],
            "thresh": [0.4, 0.8, 1.0, 1.2, 1.6, 2.0, 4.0]
        },
        "arg_builder": lambda p: ["--scale", str(p["scale"]), "--thresh", str(p["thresh"])],
        "injector": lambda args: {
            "scale": args.scale,
            "delta_pf_key_thresh": args.thresh
        }
    },
    "row_delta": {
        "grid": {
            "scale": [0.05], 
            "delta": [15],
            "row_sim": ["euclidean"],
            "divide_to": [0]
        },
        "arg_builder": lambda p: [
            "--scale", str(p["scale"]), 
            "--delta", str(p["delta"]),  
            "--row_sim", str(p["row_sim"]),
            "--divide_to", str(p["divide_to"])
        ],
        "injector": lambda args: {
            "delta_pf_key_on": 1,
            "delta_type": "row",
            "scale": args.scale,
            "delta_mlp": "Regular", 
            "row_delta_threshold": args.delta,
            "row_similarity_metric": args.row_sim,
            "divide_to": args.divide_to,
            "flash":True,
        }
    },
    "nm_delta": {
        "grid": {
            "scale": [0.05], 
            "delta": [0,0.5,1,1.5]
        },
        "arg_builder": lambda p: [
            "--scale", str(p["scale"]), 
            "--delta", str(p["delta"]), 
        ],
        "injector": lambda args: {
            "delta_type": "nm",
            "scale": args.scale,
             "row_delta_threshold": args.delta,
        }
    },
    "mlp_delta": {
        "grid": {
            "mlp_thresh": [0.1,0.2,0.3,0.4,0.6,0.7,0.8,0.9,1,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2]
            
        },
        "arg_builder": lambda p: [
            "--mlp_thresh", str(p["mlp_thresh"])
        ],
        "injector": lambda args: {
            "delta_pf_key_on": 0,              # Force attention delta OFF
            "delta_mlp": "Delta",              # Turn MLP delta ON
            "mlp_delta_threshold": args.mlp_thresh
        }
    },
    "baseline": {
        "injector": lambda args: {
            "delta_pf_key_on": 0,              
            "delta_mlp": "Regular",  
            "flash": True,             
        }
    },
    "combined_delta": {
        "grid": {
            "scale": [0.05],
            "thresh": [0.5, 1.0, 1.5],         # Attention thresholds
            "mlp_thresh": [0.5, 1.0, 1.5]      # MLP thresholds
        },
        "arg_builder": lambda p: [
            "--scale", str(p["scale"]),
            "--thresh", str(p["thresh"]),
            "--mlp_thresh", str(p["mlp_thresh"])
        ],
        "injector": lambda args: {
            "delta_pf_key_on": 1,  
            "flash": True,                # Attention delta ON
            "delta_mlp": "Delta",                  # MLP delta ON
            "delta_pf_key_thresh": args.thresh,    # Injected from --thresh
            "mlp_delta_threshold": args.mlp_thresh,# Injected from --mlp_thresh
            "scale": args.scale
        }
    }
}

# ==========================================
# HELPER CLASSES & FUNCTIONS
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
        return super(NpEncoder, self).default(obj)

def get_next_id(directory, prefix, suffix=""):
    if not os.path.exists(directory):
        return 1
    existing_ids = set()
    pattern = re.compile(rf"^{re.escape(prefix)}(\d+){re.escape(suffix)}$")
    for item in os.listdir(directory):
        match = pattern.match(item)
        if match:
            existing_ids.add(int(match.group(1)))
    next_id = 1
    while next_id in existing_ids:
        next_id += 1
    return next_id

def get_or_create_config_dir(base_exp_dir, delta_config, force_dir_name=None):
    if not os.path.exists(base_exp_dir):
        os.makedirs(base_exp_dir)

    # If a specific directory name is requested (e.g., config_baseline), use it directly
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
                        loaded_config = json.load(f)
                    
                    is_match = True
                    for k, v in delta_config.items():
                        if k not in loaded_config or loaded_config[k] != v:
                            is_match = False
                            break
                    
                    if is_match:
                        print(f"[Config] Found existing directory: {dir_path}", flush=True)
                        return dir_path
                except Exception:
                    pass

    next_id = get_next_id(base_exp_dir, prefix="config_")
    new_dir_path = os.path.join(base_exp_dir, f"config_{next_id}")
    os.makedirs(os.path.join(new_dir_path, "tasks"), exist_ok=True)
    
    with open(os.path.join(new_dir_path, "config_params.json"), 'w') as f:
        json.dump(delta_config, f, indent=4, cls=NpEncoder)
        
    print(f"[Config] Created new directory: {new_dir_path}", flush=True)
    return new_dir_path


def save_single_task_result(config_dir, task_name, result_data):
    tasks_dir = os.path.join(config_dir, "tasks")
    file_path = os.path.join(tasks_dir, f"{task_name}.json")
    lock_path = file_path + ".lock"

    # Simple spin lock
    while os.path.exists(lock_path):
        time.sleep(0.2)

    try:
        # Acquire lock
        with open(lock_path, 'w') as f: pass

        # Overwrite the file with the full payload every time
        with open(file_path, 'w') as f:
            json.dump(result_data, f, indent=4, cls=NpEncoder)
        print(f"[IO] Saved full result to {os.path.basename(file_path)}", flush=True)

    finally:
        # Release lock
        if os.path.exists(lock_path):
            os.remove(lock_path)


def run_single_pass(lm_model, task, eval_config, glob_settings, time_internal_setting):
    """
    Runs a single evaluation pass for a task with a specific timing setting.
    Returns the results dictionary and the total wall-clock time.
    """
    print(f"    - Setting time_internal: {time_internal_setting}", flush=True)
    setattr(globVR, 'time_internal', time_internal_setting)

    # Reset Global Trackers for a clean run
    if hasattr(globVR, 'spars'): globVR.spars = 0.0
    if hasattr(globVR, 'latency_stats'): globVR.latency_stats = {}
    if hasattr(globVR, 'latency_events'): globVR.latency_events = []
    if hasattr(globVR, 'sequence_lengths'): globVR.sequence_lengths = []
    if hasattr(globVR, 'mlp_spars'): globVR.mlp_spars = 0.0
    if hasattr(globVR, 'mlp_spars_count'): globVR.mlp_spars_count = 0
    
    torch.cuda.empty_cache()
    gc.collect()

    start_time = time.time()
    try:
        eval_output = simple_evaluate(
            model=lm_model,   
            tasks=[task],
            num_fewshot=eval_config["shot"],
            limit=eval_config["limit"],
            apply_chat_template=True
        )
    except Exception as e:
        print(f"    - [Error] simple_evaluate failed for task {task}: {e}", flush=True)
        return None, 0
    end_time = time.time()
    total_time = end_time - start_time
    print(f"    - Task '{task}' completed in {total_time:.2f} seconds.", flush=True)
    
    glob_set.resolve_latency_events()

    # Process latency stats from globVR
    latency_breakdown = {}
    if hasattr(globVR, 'latency_stats'):
        for metric, stats in globVR.latency_stats.items():
            if stats['calls'] > 0:
                avg_ms = stats['time_ms'] / stats['calls']
                latency_breakdown[metric] = avg_ms
    
    total_avg_ms = latency_breakdown.get('time_forward_total', 0.0)

    # Calculate sequence length statistics
    seq_lengths = getattr(globVR, 'sequence_lengths', [])
    
    benchmark_stats = {
        "num_passages": len(seq_lengths),
        "seq_len_avg": 0.0,
        "seq_len_min": 0,
        "seq_len_max": 0,
        "seq_len_std": 0.0
    }

    if seq_lengths:
        benchmark_stats["seq_len_avg"] = float(np.mean(seq_lengths))
        benchmark_stats["seq_len_min"] = int(np.min(seq_lengths))
        benchmark_stats["seq_len_max"] = int(np.max(seq_lengths))
        benchmark_stats["seq_len_std"] = float(np.std(seq_lengths))

    current_sparsity = getattr(globVR, 'spars', 0.0)
    mlp_sparsity = getattr(globVR, 'mlp_spars', 0.0)
    raw_metrics = eval_output["results"].get(task, {})
    
    primary_acc = 0.0

    # 2. Explicitly handle LongBench tasks
    if "longbench" in task:
        # Priority for LongBench is F1 for QA or RougeL for Summarization
        primary_acc = (
            raw_metrics.get("qa_f1_score,none") or 
            raw_metrics.get("summary_rouge_l,none") or
            raw_metrics.get("rouge_score,none") or
            raw_metrics.get("f1,none") or
            0.0
        )
    else:
        # 3. Handle standard tasks (ARC, MMLU, etc.)
        primary_acc = (
            raw_metrics.get("acc_norm,none") or 
            raw_metrics.get("acc,none") or 
            raw_metrics.get("exact_match,none") or
            0.0
        )
        
    result_data = {
        "task_name": task,
        "timestamp": datetime.now().isoformat(),
        "shot": eval_config["shot"],
        "sparsity": current_sparsity,
        "mlp_spars": mlp_sparsity,
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
    # 1. Base Settings - Hardcoded model IDs based on flag
    torch.set_grad_enabled(False)
    exp_def = EXPERIMENT_DEFINITIONS[args.experiment_type]
    glob_settings = exp_def["injector"](args)
    
    model_id = "microsoft/bitnet-b1.58-2B-4T" if args.bitnet else "meta-llama/Meta-Llama-3.1-8B-Instruct"

    
    # Dynamically adjust max_length depending on whether we are running LongBench

    max_len = 8000 if (args.long_bench or args.all_bench) else 4096


    model_args= f"pretrained={model_id},trust_remote_code=True,dtype=bfloat16,attn_implementation=eager"
    if not glob_settings.get("flash", False):
        model_args = f"pretrained={model_id},trust_remote_code=True,dtype=bfloat16,attn_implementation=eager,max_length={max_len}"

    eval_config = {
        "shot": args.shot, 
        "limit": 100, 
        "batch_size": 1, 
        "model_args": model_args
    }


    # 2. Get Experiment Specifics (BUT DO NOT APPLY THEM YET)
    if args.experiment_type not in EXPERIMENT_DEFINITIONS:
        print(f"[Fatal] Unknown experiment type: {args.experiment_type}", flush=True)
        return

    

    # Force safety variables during model load to prevent dummy-pass OOMs
    print("[Worker] Setting safe globVR defaults for model initialization...", flush=True)
    setattr(globVR, 'delta_pf_key_on', 0)
    setattr(globVR, 'delta_mlp', 'Regular')

    # 3. Task Selection Logic
    SHORT_TASKS = [
        "triviaqa", "mmlu", "winogrande"
    ]

    LONGBENCH_TASKS = [
        "longbench_multifieldqa_en", 
        "longbench_hotpotqa", 
        "longbench_gov_report"
    ]

    if args.all_bench:
        tasks = SHORT_TASKS + LONGBENCH_TASKS
        print(f"[Worker] '--all_bench' triggered. Running {len(tasks)} total tasks.", flush=True)
    elif args.long_bench:
        tasks = LONGBENCH_TASKS
        print(f"[Worker] '--long_bench' triggered. Running {len(tasks)} LongBench tasks.", flush=True)
    elif args.short_bench:
        tasks = SHORT_TASKS
        print(f"[Worker] '--short_bench' triggered. Running {len(tasks)} short tasks.", flush=True)
    else:
        # Fallback to shot-based logic if no specific benchmark suite flag is provided
        if args.shot == 0:
            tasks = ["arc_easy", "arc_challenge", "openbookqa", "boolq", "hellaswag", "piqa", "winogrande"]
        elif args.shot == 5:
            tasks = ["triviaqa"]
        elif args.shot == 10:
            tasks = ["commonsense_qa"]
        else:
            tasks = ["arc_challenge"]

    # 4. Load Model IN BASELINE STATE
    print(f"[Worker] Loading Model: {model_id}...", flush=True)
    try:
        model_class = lm_eval.api.registry.get_model("hf")
        lm_model = model_class.create_from_arg_string(
            eval_config["model_args"], 
            {
                "batch_size": eval_config["batch_size"],
                "device": "cuda" # <--- ADD THIS BACK
            }
        )
    except Exception as e:
        print(f"[Fatal Worker Error] Model load failed: {e}", flush=True)
        return

    # 5. NOW INJECT THE REAL GLOBALS
    print(f"[Worker] Model loaded safely. Applying {args.experiment_type} settings: {glob_settings}", flush=True)
    for key, value in glob_settings.items():
        setattr(globVR, key, value)

    # 6. Run Tasks
    # Force the directory name if we are running the baseline
    force_name = "config_baseline" if args.experiment_type == "baseline" else None
    config_dir = get_or_create_config_dir(experiment_dir, glob_settings, force_dir_name=force_name)
    
    for task in tasks:
        print(f"--- Running Task: {task} ---", flush=True)
        print(f"  -> Pass: Measuring total benchmark time AND internal latency breakdown (internal syncs ON)", flush=True)

        # Hardcode time_internal_setting to True
        results, total_time = run_single_pass(lm_model, task, eval_config, glob_settings, time_internal_setting=True)

        if results:
            final_result_data = results
            final_result_data['total_benchmark_time_s'] = total_time
            final_result_data["parameters"] = glob_settings
            final_result_data["experiment_type"] = args.experiment_type
            
            print(f"Task: {task} | Acc: {final_result_data.get('accuracy', 0.0):.4f} | Total Time: {total_time:.2f}s", flush=True)
            
            # Print the JSON to stdout so you always see the output in the console
            print(json.dumps(final_result_data, indent=4, cls=NpEncoder), flush=True)
            
            save_single_task_result(config_dir, task, final_result_data)
        else:
            print(f"[Error] No results generated for task {task}.", flush=True)

        gc.collect()
        torch.cuda.empty_cache()

    print("[Worker] Finished. Exiting.", flush=True)
# ==========================================
# MAIN ENTRY POINT
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # Model Selection Flags (Mutually Exclusive)
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument('--bitnet', action='store_true', help="Run evaluation using BitNet model")
    model_group.add_argument('--llama', action='store_true', help="Run evaluation using LLaMA model")

    # Core Args
    parser.add_argument('--mode', type=str, default='master', choices=['master', 'worker'])
    parser.add_argument('--experiment_type', type=str, required=True, default='row_delta', choices=EXPERIMENT_DEFINITIONS.keys())
    parser.add_argument('--shot', default=0, type=int)
    
    # Custom Name Parameter
    parser.add_argument('--exp_name', default=None, type=str, help="Custom folder name for the experiment output")
    parser.add_argument('--conf_name', default=None, type=str)
    
    # Benchmark Suite Selection (Mutually Exclusive)
    bench_group = parser.add_mutually_exclusive_group()
    bench_group.add_argument('--short_bench', action='store_true', help="Run standard short-context tasks")
    bench_group.add_argument('--long_bench', action='store_true', help="Run LongBench tasks only")
    bench_group.add_argument('--all_bench', action='store_true', help="Run ALL tasks (Short + LongBench)")
    
    # Shared Experiment Args
    parser.add_argument('--scale', default=0.05, type=float)
    parser.add_argument('--thresh', default=0.6, type=float)
    parser.add_argument('--mlp_thresh', default=0.0, type=float)
    parser.add_argument('--delta', default=1.0, type=float)
    parser.add_argument('--row_sim', default="cos", type=str)
    parser.add_argument('--divide_to', default=1, type=int)
    
    args = parser.parse_args()

    # Determine Base Storage Path
    model_arch = "bitnet" if args.bitnet else "llama"
    base_storage_path = os.path.join("snellius_experiments", model_arch)

    # Ensure base storage path exists
    os.makedirs(base_storage_path, exist_ok=True)

    # -----------------------------------------------------------
    # MASTER MODE
    # -----------------------------------------------------------
    if args.mode == 'master':
        
        # Auto-generate a name if none was provided
        if args.exp_name is None:
            auto_id = get_next_id(base_storage_path, prefix=f"experiment_{args.experiment_type}_")
            args.exp_name = f"experiment_{args.experiment_type}_{auto_id}"
        
        exp_def = EXPERIMENT_DEFINITIONS[args.experiment_type]
        grid_params = exp_def.get("grid", {})
        
        import itertools
        keys = list(grid_params.keys()) if grid_params else []
        values = list(grid_params.values()) if grid_params else []
        param_grid = list(itertools.product(*values)) if grid_params else [{}]
        
        print(f"[Master] Starting {args.experiment_type} Grid Search for {model_arch.upper()}.", flush=True)
        print(f"[Master] Parameters: {keys}", flush=True)
        print(f"[Master] Total Experiments: {len(param_grid)}", flush=True)
        print(f"[Master] Output Folder: {args.exp_name}", flush=True)

        for i, combo in enumerate(param_grid):
            current_params = dict(zip(keys, combo)) if keys else {}
            
            print(f"\n=== Step {i+1}/{len(param_grid)}: {current_params} ===", flush=True)
            
            # Spawn ONE worker per config
            cmd = [
                sys.executable, sys.argv[0], "--mode", "worker",
                "--experiment_type", args.experiment_type, 
                "--exp_name", args.exp_name,
                "--shot", str(args.shot)
            ]
            
            # Pass Model Selection Flags
            if args.bitnet: cmd.append("--bitnet")
            if args.llama: cmd.append("--llama")
                
            # Pass Benchmark Flags
            if args.short_bench: cmd.append("--short_bench")
            if args.long_bench: cmd.append("--long_bench")
            if args.all_bench: cmd.append("--all_bench")
            
            if "arg_builder" in exp_def:
                cmd.extend(exp_def["arg_builder"](current_params))
            
            try:
                print(f"[Master] Running worker for {current_params}", flush=True)
                # Does NOT capture output, streams directly to your terminal
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError as e:
                print(f"[Master] Worker failed for {current_params}. Continuing...", flush=True)

    # -----------------------------------------------------------
    # WORKER MODE
    # -----------------------------------------------------------
    elif args.mode == 'worker':
        # Safely handle the case where exp_name wasn't passed to a direct worker call
        if args.exp_name is None:
             args.exp_name = "default_worker_run"

        experiment_dir = os.path.join(base_storage_path, args.exp_name)
        
        run_worker_process(args, experiment_dir)