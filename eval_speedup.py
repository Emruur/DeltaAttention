import os
import sys
import json
import argparse
import torch
import numpy as np
import re
import gc
import time
import itertools
import subprocess
from datetime import datetime

# ==========================================
# IMPORTS & REGISTRATION
# ==========================================
try:
    import globVR
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    from modeling_bitnet import BitNetForCausalLM, BitNetConfig
    from datasets import load_dataset
    import glob_set 
    
    # Register Model and Config
    AutoConfig.register("bitnet", BitNetConfig, exist_ok=True)
    AutoModelForCausalLM.register(BitNetConfig, BitNetForCausalLM, exist_ok=True)
except ImportError:
    pass 

# ==========================================
# EXPERIMENT DEFINITIONS
# ==========================================
EXPERIMENT_DEFINITIONS = {
    "row_delta": {
        "grid": {
            "scale": [0.05], 
            "delta": [0,20, 30],
            "row_sim": ["euclidean"]
        },
        "arg_builder": lambda p: [
            "--scale", str(p["scale"]), 
            "--delta", str(p["delta"]), 
            "--row_sim", str(p["row_sim"])
        ],
        "injector": lambda args: {
            "use_row_delta": True,
            "scale": args.scale,
            "row_delta_threshold": args.delta,
            "row_similarity_metric": args.row_sim
        }
    }
}

# ==========================================
# HELPER FUNCTIONS
# ==========================================
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, torch.Tensor):
            return obj.item() if obj.numel() == 1 else obj.tolist()
        return super(NpEncoder, self).default(obj)

def get_next_id(directory, prefix):
    if not os.path.exists(directory):
        return 1
    existing_ids = set()
    pattern = re.compile(rf"^{re.escape(prefix)}(\d+)$")
    for item in os.listdir(directory):
        match = pattern.match(item)
        if match:
            existing_ids.add(int(match.group(1)))
    next_id = 1
    while next_id in existing_ids:
        next_id += 1
    return next_id

def get_unique_filename(directory, prefix="conf"):
    """
    Finds the next available filename (e.g., conf1.json, conf2.json)
    to prevent overwriting results in the same experiment folder.
    """
    counter = 1
    while True:
        filename = f"{prefix}{counter}.json"
        full_path = os.path.join(directory, filename)
        if not os.path.exists(full_path):
            return filename
        counter += 1

def get_real_text_input(tokenizer, seq_len, device):
    """
    Fetches real text from WikiText-2 to ensure realistic sparsity patterns.
    """
    print("[Data] Loading WikiText-2 validation set...")
    try:
        # Load a small slice of wikitext
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
        
        # Concatenate text until we have enough tokens
        text = ""
        for item in dataset:
            text += item["text"]
            if len(text) > seq_len * 10: # Crude approximation (chars vs tokens)
                break
                
        # Tokenize
        encodings = tokenizer(text, return_tensors="pt")
        
        # Ensure we have enough tokens
        if encodings.input_ids.shape[1] < seq_len:
             print("[Warning] Text too short, repeating content.")
             input_ids = encodings.input_ids.repeat(1, 2)[:, :seq_len].to(device)
        else:
             input_ids = encodings.input_ids[:, :seq_len].to(device)
        
        print(f"[Data] Loaded real text block of shape: {input_ids.shape}")
        return input_ids
    except Exception as e:
        print(f"[Warning] Failed to load WikiText ({e}). Falling back to random noise.")
        return torch.randint(0, 1000, (1, seq_len)).to(device)

# [Inside speedup_evaluator.py]

def benchmark_latency(model, input_ids, warmup=5, repeats=20):
    
    print(f"   [Bench] Warming up ({warmup} iters)...")
    with torch.no_grad():
        for _ in range(warmup):
            model(input_ids)
    
    # 2. Reset Global Timers (Empty the dictionary entirely)
    globVR.latency_stats = {}
    globVR.spars = 0.0 
    
    print(f"   [Bench] Measuring ({repeats} iters)...")
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t_start_total = time.time()
    
    with torch.no_grad():
        for i in range(repeats):
            model(input_ids)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            if i % 5 == 0:
                torch.cuda.empty_cache()
            
    t_end_total = time.time()
    print(f"   [Bench] Total Wall Clock (All Layers): {t_end_total - t_start_total:.4f}s")
            
    # 3. Calculate Stats from Dictionary
    latency_breakdown = {}
    if hasattr(globVR, 'latency_stats'):
        for metric, stats in globVR.latency_stats.items():
            if stats['calls'] > 0:
                # Calculate average and store in the breakdown dictionary
                avg_ms = stats['time_ms'] / stats['calls']
                latency_breakdown[metric] = avg_ms
                
    avg_sparsity = getattr(globVR, 'spars', 0.0)
    
    return latency_breakdown, avg_sparsity

# --------------------------------------------------------------------------

# ==========================================
def run_worker_process(args, experiment_dir):
    # 1. Setup Model & Tokenizer
    print(f"[Worker] Loading Model & Tokenizer...")
    model_id = "microsoft/bitnet-b1.58-2B-4T"
    
    config = AutoConfig.from_pretrained(model_id, trust_remote_code=False)
    config.attn_implementation = "eager" 
    
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=False)
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        config=config,
        trust_remote_code=False,
        torch_dtype=torch.bfloat16
    ).to("cuda")
    model.eval()

    # 2. Get Real Data
    # We load this BEFORE applying settings so we have a clean baseline input
    input_ids = get_real_text_input(tokenizer, args.seq_len, "cuda")

    # 3. Apply Experiment Settings
    if args.experiment_type not in EXPERIMENT_DEFINITIONS:
        print(f"[Fatal] Unknown experiment type: {args.experiment_type}")
        return

    exp_def = EXPERIMENT_DEFINITIONS[args.experiment_type]
    glob_settings = exp_def["injector"](args)
    
    print(f"[Worker] Applying settings: {glob_settings}")
    for key, value in glob_settings.items():
        setattr(globVR, key, value)

    # 4. Run Benchmark
    os.makedirs(experiment_dir, exist_ok=True)
    print(f"--- Running Speed Benchmark (Seq Len: {args.seq_len}) ---")
    
    gc.collect()
    torch.cuda.empty_cache()
    
    try:
        # benchmark_latency now returns a dictionary of latencies
        latency_breakdown, avg_sparsity = benchmark_latency(
            model, 
            input_ids,
            repeats=20 
        )
        
        # 5. Save Results
        result_data = {
            "timestamp": datetime.now().isoformat(),
            "experiment_type": args.experiment_type,
            "parameters": glob_settings,
            "seq_len": args.seq_len,
            "latency_breakdown_ms": latency_breakdown, # <--- The nested dictionary
            "sparsity": avg_sparsity
        }
        
        # Generate unique filename: conf1.json, conf2.json, etc.
        filename = get_unique_filename(experiment_dir, prefix="conf")
        file_path = os.path.join(experiment_dir, filename)
        
        with open(file_path, 'w') as f:
            json.dump(result_data, f, indent=4, cls=NpEncoder)
            
        print(f"[Result] Saved to {filename}")
        print(json.dumps(latency_breakdown, indent=2)) # Print the breakdown cleanly to console

    except Exception as e:
        print(f"[Error] Benchmark failed: {e}")
        import traceback
        traceback.print_exc()

# ==========================================
# MAIN ENTRY POINT
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='master', choices=['master', 'worker'])
    parser.add_argument('--experiment_type', type=str, required=True, choices=EXPERIMENT_DEFINITIONS.keys())
    parser.add_argument('--exp_num', default=None, type=int)
    parser.add_argument('--seq_len', default=1024, type=int)
    
    # Shared Args
    parser.add_argument('--scale', default=0.05, type=float)
    parser.add_argument('--thresh', default=0.6, type=float)
    parser.add_argument('--delta', default=1.0, type=float)
    parser.add_argument('--row_sim', default="cos", type=str)

    args = parser.parse_args()

    # -----------------------------------------------------------
    # MASTER MODE
    # -----------------------------------------------------------
    if args.mode == 'master':
        base_storage_path = "speedup_experiments"
        
        if args.exp_num is None:
            args.exp_num = get_next_id(base_storage_path, prefix=f"speed_{args.experiment_type}_")
        
        exp_def = EXPERIMENT_DEFINITIONS[args.experiment_type]
        grid_params = exp_def["grid"]
        
        keys = list(grid_params.keys())
        values = list(grid_params.values())
        param_grid = list(itertools.product(*values))
        
        print(f"[Master] Starting Speed Benchmark (Real Text) for {args.experiment_type}")
        print(f"[Master] ID: {args.exp_num}")

        for i, combination in enumerate(param_grid):
            current_params = dict(zip(keys, combination))
            print(f"\n=== Step {i+1}/{len(param_grid)}: {current_params} ===")
            
            cmd = [
                sys.executable, sys.argv[0],
                "--mode", "worker",
                "--experiment_type", args.experiment_type,
                "--exp_num", str(args.exp_num),
                "--seq_len", str(args.seq_len)
            ]
            cmd.extend(exp_def["arg_builder"](current_params))
            
            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError:
                print(f"[Master] Worker failed. Continuing...")

    # -----------------------------------------------------------
    # WORKER MODE
    # -----------------------------------------------------------
    elif args.mode == 'worker':
        base_storage_path = "speedup_experiments"
        exp_folder_name = f"speed_{args.experiment_type}_{args.exp_num}"
        experiment_dir = os.path.join(base_storage_path, exp_folder_name)
        
        run_worker_process(args, experiment_dir)