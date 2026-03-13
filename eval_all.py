import os
import sys
import json
import argparse
import torch
import numpy as np
import re
import gc
import itertools
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
            "delta": [0,15,20],
            "row_sim": ["euclidean"]
        },
        "arg_builder": lambda p: [
            "--scale", str(p["scale"]), 
            "--delta", str(p["delta"]), 
            "--row_sim", str(p["row_sim"])
        ],
        "injector": lambda args: {
            "delta_type": "row",
            "scale": args.scale,
            "row_delta_threshold": args.delta,
            "row_similarity_metric": args.row_sim
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
            "mlp_thresh": [ 0.6, 0.7, 0.8, 0.9] # Dedicated sweep for MLP threshold
        },
        "arg_builder": lambda p: [
            "--mlp_thresh", str(p["mlp_thresh"]), 
        ],
        "injector": lambda args: {
            "delta_pf_key_on": 0,              # Force attention delta OFF
            "delta_mlp": "Delta",              # Turn MLP delta ON
            "mlp_delta_threshold": args.mlp_thresh # Set the current MLP threshold
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

def get_or_create_config_dir(base_exp_dir, delta_config):
    if not os.path.exists(base_exp_dir):
        os.makedirs(base_exp_dir)

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

def save_single_task_result(config_dir, task_name, result_data):
    tasks_dir = os.path.join(config_dir, "tasks")
    base_filename = f"{task_name}.json"
    file_path = os.path.join(tasks_dir, base_filename)
    
    if os.path.exists(file_path):
        counter = 1
        while True:
            new_path = os.path.join(tasks_dir, f"{task_name}_{counter}.json")
            if not os.path.exists(new_path):
                file_path = new_path
                break
            counter += 1
    
    with open(file_path, 'w') as f:
        json.dump(result_data, f, indent=4, cls=NpEncoder)
    print(f"[IO] Saved {os.path.basename(file_path)}")

# ==========================================
# WORKER FUNCTION
# ==========================================
def run_worker_process(args, experiment_dir):
    # 1. Base Settings
    eval_config = {
        "shot": args.shot,
        "limit": 100,
        "batch_size": 1,
        "device": "cuda",
        "model_args": "pretrained=microsoft/bitnet-b1.58-2B-4T,trust_remote_code=False,dtype=bfloat16,attn_implementation=eager"
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

    # 3. Determine Tasks
    if args.shot == 0:
        tasks = ["arc_easy", "arc_challenge", "openbookqa", "boolq", "hellaswag", "piqa", "winogrande"]
    elif args.shot == 5:
        tasks = ["triviaqa", "mmlu"]
        tasks = ["triviaqa"]
    elif args.shot == 10:
        tasks = ["commonsense_qa", "truthfulqa_mc2"]
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
    config_dir = get_or_create_config_dir(experiment_dir, glob_settings)
    
    for task in tasks:
        print(f"--- Running Task: {task} ---")
        
        # Reset Global Trackers for a clean run per task
        if hasattr(globVR, 'spars'): globVR.spars = 0.0
        if hasattr(globVR, 'total_attn_time'): globVR.total_attn_time = 0.0
        if hasattr(globVR, 'total_attn_calls'): globVR.total_attn_calls = 0
        if hasattr(globVR, 'mlp_spars'): globVR.mlp_spars = 0.0
        if hasattr(globVR, 'mlp_spars_count'): globVR.mlp_spars_count = 0
        
        # Aggressive cleaning before run
        torch.cuda.empty_cache()
        gc.collect()

        try:
            eval_output = simple_evaluate(
                model=lm_model,   
                tasks=[task],
                num_fewshot=eval_config["shot"],
                limit=eval_config["limit"]
            )

            avg_time = 0.0
            if hasattr(globVR, 'total_attn_calls') and globVR.total_attn_calls > 0:
                avg_time = globVR.total_attn_time / globVR.total_attn_calls

            current_sparsity = getattr(globVR, 'spars', 0.0)
            mlp_sparsity = getattr(globVR, 'mlp_spars', 0.0)
            raw_metrics = eval_output["results"].get(task, {})

            primary_acc = raw_metrics.get("acc,none") or raw_metrics.get("acc_norm,none") or raw_metrics.get("acc") or raw_metrics.get("exact_match,remove_whitespace")  or 0.0

            result_data = {
                "task_name": task,
                "timestamp": datetime.now().isoformat(),
                "experiment_type": args.experiment_type,
                "parameters": glob_settings, 
                "shot": eval_config["shot"],
                "sparsity": current_sparsity,
                "mlp_spars": mlp_sparsity,
                "avg_attn_latency_ms": avg_time,
                "accuracy": primary_acc,
                "metrics": raw_metrics
            }
            
            print(f"Task: {task} | Attn Sparsity: {current_sparsity:.4f} | MLP Sparsity: {mlp_sparsity:.4f} | Acc: {primary_acc:.4f}")
            save_single_task_result(config_dir, task, result_data)
            
            # Aggressive cleanup AFTER run
            del eval_output
            del raw_metrics
            gc.collect()
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"[Error] Task {task} failed: {e}")

    print("[Worker] Finished. Exiting.")

# ==========================================
# MAIN ENTRY POINT
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # Core Args
    parser.add_argument('--mode', type=str, default='master', choices=['master', 'worker'])
    parser.add_argument('--experiment_type', type=str, required=True, choices=EXPERIMENT_DEFINITIONS.keys())
    parser.add_argument('--shot', default=0, type=int)
    parser.add_argument('--exp_num', default=None, type=int)
    parser.add_argument('--conf_name', default=None, type=str)
    
    # Shared Experiment Args
    parser.add_argument('--scale', default=0.05, type=float)
    
    # Scale_Delta specific (Attention)
    parser.add_argument('--thresh', default=0.6, type=float)
    
    # MLP_Delta specific
    parser.add_argument('--mlp_thresh', default=0.0, type=float)

    # Row_Delta specific
    parser.add_argument('--delta', default=1.0, type=float)
    parser.add_argument('--row_sim', default="cos", type=str)
    
    args = parser.parse_args()

    # -----------------------------------------------------------
    # MASTER MODE
    # -----------------------------------------------------------
    if args.mode == 'master':
        base_storage_path = "experiments"
        
        if args.exp_num is None:
            args.exp_num = get_next_id(base_storage_path, prefix=f"experiment_{args.experiment_type}_")
        
        exp_def = EXPERIMENT_DEFINITIONS[args.experiment_type]
        grid_params = exp_def["grid"]
        
        keys = list(grid_params.keys())
        values = list(grid_params.values())
        param_grid = list(itertools.product(*values))
        
        print(f"[Master] Starting {args.experiment_type} Grid Search.")
        print(f"[Master] Parameters: {keys}")
        print(f"[Master] Total Experiments: {len(param_grid)}")
        print(f"[Master] Experiment ID: {args.exp_num}")

        for i, combination in enumerate(param_grid):
            current_params = dict(zip(keys, combination))
            
            print(f"\n=== Step {i+1}/{len(param_grid)}: {current_params} ===")
            
            cmd = [
                sys.executable, sys.argv[0],
                "--mode", "worker",
                "--experiment_type", args.experiment_type,
                "--exp_num", str(args.exp_num),
                "--shot", str(args.shot)
            ]
            
            # The lambda builder pulls the right args
            cmd.extend(exp_def["arg_builder"](current_params))
            
            if args.conf_name:
                cmd.extend(["--conf_name", args.conf_name])

            try:
                # Passing the modified env to the subprocess
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError as e:
                print(f"[Master] Worker failed for {current_params}. Continuing...")

    # -----------------------------------------------------------
    # WORKER MODE
    # -----------------------------------------------------------
    elif args.mode == 'worker':
        base_storage_path = "experiments"
        exp_folder_name = f"experiment_{args.experiment_type}_{args.exp_num}"
        experiment_dir = os.path.join(base_storage_path, exp_folder_name)
        
        run_worker_process(args, experiment_dir)