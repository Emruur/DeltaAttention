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

import globVR
import glob_set 
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from modeling_bitnet import BitNetForCausalLM, BitNetConfig

# Register Model and Config
AutoConfig.register("bitnet", BitNetConfig, exist_ok=True)
AutoModelForCausalLM.register(BitNetConfig, BitNetForCausalLM, exist_ok=True)

# ==========================================
# EXPERIMENT DEFINITIONS
# ==========================================
EXPERIMENT_DEFINITIONS = {
    "baseline": {
        "grid": {
            "seq_len": [1024, 2048, 3072],
        },
        "arg_builder": lambda p: ["--seq_len", str(p["seq_len"])],
        "injector": lambda args: {
            "delta_pf_key_on": 0,    # Forces standard matmul path
            "use_row_delta": False,
            "collect_delta_pf_key": 0,
            "scale": 0.0  # Not used in baseline
        }
    },
    "scale_delta": {
        "grid": {
            "seq_len": [1024, 2048, 3072],
            "scale": [0.05], 
            "thresh": [1]
        },
        "arg_builder": lambda p: ["--seq_len", str(p["seq_len"]), "--scale", str(p["scale"]), "--thresh", str(p["thresh"])],
        "injector": lambda args: {
            "delta_pf_key_on": 1,
            "use_row_delta": False,
            "scale": args.scale,
            "delta_pf_key_thresh": args.thresh,
            "sink_size": 64
        }
    },
    "row_delta": {
        "grid": {
            "seq_len": [1024, 2048, 3072, 4096],
            "delta": [15,20],
            "row_sim": ["euclidean"]
        },
        "arg_builder": lambda p: [
            "--seq_len", str(p["seq_len"]),
            "--delta", str(p["delta"]), 
            "--row_sim", str(p["row_sim"])
        ],
        "injector": lambda args: {
            "delta_pf_key_on": 1,
            "use_row_delta": True,
            "row_delta_threshold": args.delta,
            "row_similarity_metric": args.row_sim,
        }
    },
    "nm_delta": {
        "grid": {
            "delta": [0,0.5,1],
            "seq_len": [1024, 2048],
        },
        "arg_builder": lambda p: [
            "--seq_len", str(p["seq_len"]),
            "--delta", str(p["delta"]), 
        ],
        "injector": lambda args: {
            "delta_type": "nm",
            "scale": args.scale,
             "row_delta_threshold": args.delta,
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
    if not os.path.exists(directory): return 1
    existing_ids = set()
    pattern = re.compile(rf"^{re.escape(prefix)}(\d+)$")
    for item in os.listdir(directory):
        match = pattern.match(item)
        if match: existing_ids.add(int(match.group(1)))
    next_id = 1
    while next_id in existing_ids: next_id += 1
    return next_id

def get_unique_filename(directory, prefix="conf"):
    counter = 1
    while True:
        filename = f"{prefix}{counter}.json"
        if not os.path.exists(os.path.join(directory, filename)): return filename
        counter += 1

def get_real_text_input(tokenizer, seq_len, device):
    from datasets import load_dataset
    try:
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
        text = "".join([item["text"] for item in dataset[:100]])
        encodings = tokenizer(text, return_tensors="pt")
        if encodings.input_ids.shape[1] < seq_len:
             return encodings.input_ids.repeat(1, (seq_len // encodings.input_ids.shape[1]) + 1)[:, :seq_len].to(device)
        return encodings.input_ids[:, :seq_len].to(device)
    except:
        return torch.randint(0, 1000, (1, seq_len)).to(device)

def benchmark_latency(model, input_ids, warmup=10, repeats=20):
    """
    Benchmarks the model and extracts internal attention stats from globVR.
    """
    # 1. Warmup: Critical to initialize CUDA kernels and settle GPU clocks
    print(f"   [Bench] Warming up ({warmup} iters)...")
    with torch.no_grad():
        for _ in range(warmup):
            model(input_ids)
    
    # 2. Reset Statistics: Ensure we don't include warmup time in our metrics
    # We clear the dictionary that your glob_set.update_latency writes to.
    globVR.latency_stats = {}
    if hasattr(globVR, 'spars'):
        globVR.spars = 0.0 
    
    # 3. Measurement Loop
    print(f"   [Bench] Measuring ({repeats} iters)...")
    torch.cuda.synchronize()
    t_start_total = time.perf_counter() # Use high-resolution counter
    
    with torch.no_grad():
        for i in range(repeats):
            model(input_ids)
            # Sync after every pass to ensure hardware finished the work
            torch.cuda.synchronize()
            
    t_end_total = time.perf_counter()
    
    # 4. Process Results
    total_wall_clock_ms = ((t_end_total - t_start_total) / repeats) * 1000
    
    latency_breakdown = {}
    # Extract the stats that were populated during the 'repeats' loop
    if hasattr(globVR, 'latency_stats'):
        for metric, stats in globVR.latency_stats.items():
            if stats['calls'] > 0:
                # Calculate average ms per call
                # Note: your update_latency already multiplied by 1000
                avg_ms = stats['time_ms'] / stats['calls']
                latency_breakdown[metric] = avg_ms
                
    avg_sparsity = getattr(globVR, 'spars', 0.0)
    
    # Logging for console feedback
    print(f"   [Bench] Avg Total: {total_wall_clock_ms:.2f}ms")
    if 'time_forward_total' in latency_breakdown:
        print(f"   [Bench] Avg Attn:  {latency_breakdown['time_forward_total']:.2f}ms")
            
    return latency_breakdown, total_wall_clock_ms, avg_sparsity

# ==========================================
# WORKER PROCESS
# ==========================================
def run_worker_process(args, experiment_dir):
    model_id = "microsoft/bitnet-b1.58-2B-4T"
    config = AutoConfig.from_pretrained(model_id)
    config.attn_implementation = "eager" 
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id, config=config, torch_dtype=torch.bfloat16
    ).to("cuda").eval()

    input_ids = get_real_text_input(tokenizer, args.seq_len, "cuda")

    # Inject Parameters
    exp_def = EXPERIMENT_DEFINITIONS[args.experiment_type]
    glob_settings = exp_def["injector"](args)
    for key, value in glob_settings.items():
        setattr(globVR, key, value)

    try:

        latency_breakdown, total_avg, avg_sparsity = benchmark_latency(model, input_ids)
        
        result_data = {
            "timestamp": datetime.now().isoformat(),
            "experiment_type": args.experiment_type,
            "parameters": glob_settings,
            "seq_len": args.seq_len,
            "total_avg_ms": total_avg,
            "latency_breakdown_ms": latency_breakdown,
            "avg_sparsity": avg_sparsity

        }
        
        os.makedirs(experiment_dir, exist_ok=True)
        filename = get_unique_filename(experiment_dir)
        with open(os.path.join(experiment_dir, filename), 'w') as f:
            json.dump(result_data, f, indent=4, cls=NpEncoder)
            
        print(f"[Result] SeqLen {args.seq_len}: {total_avg:.2f}ms")
    except Exception as e:
        print(f"[Error] {e}")

# ==========================================
# MAIN
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='master', choices=['master', 'worker'])
    parser.add_argument('--experiment_type', type=str, required=True, choices=EXPERIMENT_DEFINITIONS.keys())
    parser.add_argument('--exp_num', default=None, type=int)
    parser.add_argument('--seq_len', default=1024, type=int)
    parser.add_argument('--scale', default=0.05, type=float)
    parser.add_argument('--thresh', default=0.6, type=float)
    parser.add_argument('--delta', default=1.0, type=float)
    parser.add_argument('--row_sim', default="cos", type=str)

    args = parser.parse_args()

    if args.mode == 'master':
        base_path = "speedup_experiments"
        if args.exp_num is None:
            args.exp_num = get_next_id(base_path, f"speed_{args.experiment_type}_")
        
        exp_def = EXPERIMENT_DEFINITIONS[args.experiment_type]
        keys, values = zip(*exp_def["grid"].items())
        param_grid = [dict(zip(keys, v)) for v in itertools.product(*values)]
        
        print(f"[Master] Starting {args.experiment_type} (ID: {args.exp_num})")

        for i, params in enumerate(param_grid):
            print(f"\n=== Progress {i+1}/{len(param_grid)} | {params} ===")
            cmd = [sys.executable, sys.argv[0], "--mode", "worker", 
                   "--experiment_type", args.experiment_type, "--exp_num", str(args.exp_num)]
            cmd.extend(exp_def["arg_builder"](params))
            subprocess.run(cmd, check=True)

    elif args.mode == 'worker':
        exp_folder = f"speed_{args.experiment_type}_{args.exp_num}"
        run_worker_process(args, os.path.join("speedup_experiments", exp_folder))