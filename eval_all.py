import os
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
import lm_eval.api.registry
from lm_eval.evaluator import simple_evaluate
from transformers import AutoConfig, AutoModelForCausalLM
from modeling_bitnet import BitNetForCausalLM, BitNetConfig

# Register Model and Config
AutoConfig.register("bitnet", BitNetConfig, exist_ok=True)
AutoModelForCausalLM.register(BitNetConfig, BitNetForCausalLM, exist_ok=True)


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
            "divide_to": [1,4]
        },
        "arg_builder": lambda p: [
            "--scale", str(p["scale"]), 
            "--delta", str(p["delta"]),  
            "--row_sim", str(p["row_sim"]),
            "--divide_to", str(p["divide_to"])
        ],
        "injector": lambda args: {
            "delta_type": "row",
            "scale": args.scale,
            "delta_mlp": "Regular", 
            "row_delta_threshold": args.delta,
            "row_similarity_metric": args.row_sim,
            "divide_to": args.divide_to,
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
            "delta_pf_key_on": 1,                  # Attention delta ON
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
        print(f"[Config] Using forced directory: {dir_path}")
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
                        print(f"[Config] Found existing directory: {dir_path}")
                        return dir_path
                except Exception:
                    pass

    next_id = get_next_id(base_exp_dir, prefix="config_")
    new_dir_path = os.path.join(base_exp_dir, f"config_{next_id}")
    os.makedirs(os.path.join(new_dir_path, "tasks"), exist_ok=True)
    
    with open(os.path.join(new_dir_path, "config_params.json"), 'w') as f:
        json.dump(delta_config, f, indent=4, cls=NpEncoder)
        
    print(f"[Config] Created new directory: {new_dir_path}")
    return new_dir_path


def run_worker_process(args, experiment_dir):
    # 1. Base Settings
    eval_config = {
        "shot": args.shot, "limit": 100, "batch_size": 1, "device": "cuda",
        "model_args": "pretrained=microsoft/bitnet-b1.58-2B-4T,trust_remote_code=False,dtype=bfloat16,attn_implementation=eager",
    }

    # 2. Apply Experiment Specifics
    if args.experiment_type not in EXPERIMENT_DEFINITIONS:
        print(f"[Fatal] Unknown experiment type: {args.experiment_type}")
        return

    exp_def = EXPERIMENT_DEFINITIONS[args.experiment_type]
    glob_settings = exp_def["injector"](args)

    print(f"[Worker] Applying {args.experiment_type} settings: {glob_settings}")
    
    for key, value in glob_settings.items():
        setattr(globVR, key, value)

    all_possible_tasks = [
        "arc_easy", "arc_challenge", "openbookqa", "boolq", 
        "hellaswag", "piqa", "winogrande", "triviaqa", 
        "mmlu", "commonsense_qa", "truthfulqa_mc2"
    ]

    if getattr(args, 'run_all_tasks', False):
        tasks = all_possible_tasks
        print(f"[Worker] '--run_all_tasks' triggered. Running {len(tasks)} tasks.")
    else:
        if args.shot == 0:
            tasks = ["arc_easy", "arc_challenge", "openbookqa", "boolq", "hellaswag", "piqa", "winogrande"]
        elif args.shot == 5:
            tasks = ["triviaqa"]
        elif args.shot == 10:
            tasks = ["commonsense_qa"]
        else:
            tasks = ["arc_challenge"]

    # 4. Load Model
    print(f"[Worker] Loading Model...")
    try:
        model_class = lm_eval.api.registry.get_model("hf")
        lm_model = model_class.create_from_arg_string(
            eval_config["model_args"], 
            {
                "batch_size": eval_config["batch_size"],
                "device": eval_config["device"]
            }
        )
    except Exception as e:
        print(f"[Fatal Worker Error] Model load failed: {e}")
        return

    # 5. Run Tasks
    # Force the directory name if we are running the baseline
    force_name = "config_baseline" if args.experiment_type == "baseline" else None
    config_dir = get_or_create_config_dir(experiment_dir, glob_settings, force_dir_name=force_name)
    
    for task in tasks:
        print(f"--- Running Task: {task} ---")
        
        is_sync_setting = (args.time_internal == 'yes')
        
        if is_sync_setting:
            print(f"  -> Pass: Measuring internal latency breakdown (internal syncs ON)")
        else:
            print(f"  -> Pass: Measuring total benchmark time (internal syncs OFF)")

        results, total_time = run_single_pass(lm_model, task, eval_config, glob_settings, time_internal_setting=is_sync_setting)

        if results:
            final_result_data = results
            final_result_data['total_benchmark_time_s'] = total_time
            final_result_data["parameters"] = glob_settings
            final_result_data["experiment_type"] = args.experiment_type
            
            print(f"Task: {task} | Acc: {final_result_data.get('accuracy', 0.0):.4f} | Total Time: {total_time:.2f}s")
            save_single_task_result(config_dir, task, final_result_data)
        else:
            print(f"[Error] No results generated for task {task}.")

        gc.collect()
        torch.cuda.empty_cache()

    print("[Worker] Finished. Exiting.")


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

        # Read existing data if it exists
        existing_data = {}
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                try:
                    existing_data = json.load(f)
                except json.JSONDecodeError:
                    pass # Ignore corrupt file, will be overwritten

        # Check if the current run is the 'sync' run (has breakdown)
        is_sync_run = "timings" in result_data and result_data["timings"].get("latency_breakdown_ms")

        if is_sync_run:
            # This is the 'yes' pass. Only add its timings to the existing data.
            existing_data['timings'] = result_data.get('timings', {})
            # Also store its own benchmark time for debugging
            existing_data['sync_pass_benchmark_time_s'] = result_data.get('total_benchmark_time_s')
            final_data = existing_data
        else:
            # This is the 'no' pass. It's the primary source of data.
            # Overwrite everything EXCEPT for the timings, which might have been written by the 'yes' pass.
            timings_to_preserve = existing_data.get('timings', {})
            final_data = result_data
            final_data['timings'] = timings_to_preserve
        
        # Write back the merged data
        with open(file_path, 'w') as f:
            json.dump(final_data, f, indent=4, cls=NpEncoder)
        print(f"[IO] Saved/Merged result to {os.path.basename(file_path)}")

    finally:
        # Release lock
        if os.path.exists(lock_path):
            os.remove(lock_path)

def run_single_pass(lm_model, task, eval_config, glob_settings, time_internal_setting):
    """
    Runs a single evaluation pass for a task with a specific timing setting.
    Returns the results dictionary and the total wall-clock time.
    """
    print(f"    - Setting time_internal: {time_internal_setting}")
    setattr(globVR, 'time_internal', time_internal_setting)

    # Reset Global Trackers for a clean run
    if hasattr(globVR, 'spars'): globVR.spars = 0.0
    if hasattr(globVR, 'latency_stats'): globVR.latency_stats = {}
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
            limit=eval_config["limit"]
        )
    except Exception as e:
        print(f"    - [Error] simple_evaluate failed for task {task}: {e}")
        return None, 0
    end_time = time.time()
    total_time = end_time - start_time
    print(f"    - Task '{task}' completed in {total_time:.2f} seconds.")

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

    primary_acc = raw_metrics.get("acc,none") or raw_metrics.get("acc_norm,none") or raw_metrics.get("acc") or raw_metrics.get("exact_match,remove_whitespace") or 0.0

    result_data = {
        "task_name": task,
        "timestamp": datetime.now().isoformat(),
        "shot": eval_config["shot"],
        "sparsity": current_sparsity,
        "mlp_spars": mlp_sparsity,
        "benchmark_stats": benchmark_stats, # Inserted the new stats object here
        "accuracy": primary_acc,
        "timings": {
            "total_avg_ms": total_avg_ms,
            "latency_breakdown_ms": latency_breakdown,
        },
        "metrics": raw_metrics,
    }
    return result_data, total_time

# ==========================================
# MAIN ENTRY POINT
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # Core Args
    parser.add_argument('--mode', type=str, default='master', choices=['master', 'worker'])
    parser.add_argument('--experiment_type', type=str, required=True, default='row_delta',choices=EXPERIMENT_DEFINITIONS.keys())
    parser.add_argument('--shot', default=0, type=int)
    parser.add_argument('--exp_num', default=None, type=int)
    parser.add_argument('--conf_name', default=None, type=str)
    parser.add_argument('--time_internal', type=str, default='both', choices=['yes', 'no', 'both'])
    
    # Shared Experiment Args
    parser.add_argument('--scale', default=0.05, type=float)
    
    # Scale_Delta specific (Attention)
    parser.add_argument('--thresh', default=0.6, type=float)
    
    # MLP_Delta specific
    parser.add_argument('--mlp_thresh', default=0.0, type=float)

    # Row_Delta specific
    parser.add_argument('--delta', default=1.0, type=float)
    parser.add_argument('--row_sim', default="cos", type=str)

    parser.add_argument('--divide_to', default=1, type=int)
    parser.add_argument('--run_all_tasks', action='store_true', help="Run all predefined tasks at once")
    
    args = parser.parse_args()

    # -----------------------------------------------------------
    # MASTER MODE
    # -----------------------------------------------------------
    if args.mode == 'master':
        base_storage_path = "experiments"
        
        if args.exp_num is None:
            args.exp_num = get_next_id(base_storage_path, prefix=f"experiment_{args.experiment_type}_")
        
        exp_def = EXPERIMENT_DEFINITIONS[args.experiment_type]
        grid_params = exp_def.get("grid", {})
        
        import itertools
        keys = list(grid_params.keys()) if grid_params else []
        values = list(grid_params.values()) if grid_params else []
        param_grid = list(itertools.product(*values)) if grid_params else [{}]
        
        print(f"[Master] Starting {args.experiment_type} Grid Search.")
        print(f"[Master] Parameters: {keys}")
        print(f"[Master] Total Experiments: {len(param_grid)}")
        print(f"[Master] Experiment ID: {args.exp_num}")

        for i, combo in enumerate(param_grid):
            current_params = dict(zip(keys, combo)) if keys else {}
            
            print(f"\n=== Step {i+1}/{len(param_grid)}: {current_params} ===")
            
            time_modes_to_run = [args.time_internal]
            if args.time_internal == 'both':
                print(f"[Master] Spawning separate 'yes' and 'no' sync processes for {current_params}")
                time_modes_to_run = ['no', 'yes']

            for time_mode in time_modes_to_run:
                cmd = [
                    sys.executable, sys.argv[0], "--mode", "worker",
                    "--experiment_type", args.experiment_type, "--exp_num", str(args.exp_num),
                    "--shot", str(args.shot), "--time_internal", time_mode,
                ]
                if getattr(args, 'run_all_tasks', False):
                    cmd.append("--run_all_tasks")
                
                if "arg_builder" in exp_def:
                    cmd.extend(exp_def["arg_builder"](current_params))
                
                try:
                    print(f"[Master] Running worker with --time_internal {time_mode}")
                    subprocess.run(cmd, check=True)
                except subprocess.CalledProcessError as e:
                    print(f"[Master] Worker failed for {current_params} with time_mode={time_mode}. Continuing...")

    # -----------------------------------------------------------
    # WORKER MODE
    # -----------------------------------------------------------
    elif args.mode == 'worker':
        base_storage_path = "experiments"
        exp_folder_name = f"experiment_{args.experiment_type}_{args.exp_num}"
        experiment_dir = os.path.join(base_storage_path, exp_folder_name)
        
        run_worker_process(args, experiment_dir)