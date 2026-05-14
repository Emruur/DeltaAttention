"""
LongBench evaluation using THUDM's original evaluation protocol.
Reproduces the exact setup used in KVMerger, H2O, and other papers.

Key differences from eval_longbench.py (lm_eval-based):
  - Loads THUDM/LongBench dataset directly via HuggingFace datasets
  - Uses THUDM's per-task prompt templates
  - Applies chat template only for tasks that need it (NOT trec/triviaqa)
  - Uses THUDM's scoring functions (qa_f1, rouge-l, classification, retrieval)
  - Middle-truncates inputs to fit max_length
  - Task-specific max_new_tokens

Model:  meta-llama/Llama-2-7b-chat-hf  (4K context window)
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
import string
from collections import Counter
from datetime import datetime

import globVR
import glob_set
from datasets import load_dataset
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM

from modeling_llama import LlamaForCausalLM, LlamaConfig
from deltaDecoding import HybridCompressedCache

AutoConfig.register("llama", LlamaConfig, exist_ok=True)
AutoModelForCausalLM.register(LlamaConfig, LlamaForCausalLM, exist_ok=True)

MODEL_ID = "meta-llama/Llama-2-7b-chat-hf"
MAX_LENGTH = 4096

TASKS = [
    "gov_report",
    "multi_news",
    "narrativeqa",
    "qasper",
    "multifieldqa_en",
    "trec",
    "2wikimqa",
    "triviaqa",
    "passage_retrieval_en",
]

DATASET2PROMPT = {
    "narrativeqa": "You are given a story, which can be either a novel or a movie script, and a question. Answer the question asconcisely as you can, using a single phrase if possible. Do not provide any explanation.\n\nStory: {context}\n\nNow, answer the question based on the story asconcisely as you can, using a single phrase if possible. Do not provide any explanation.\n\nQuestion: {input}\n\nAnswer:",
    "qasper": "You are given a scientific article and a question. Answer the question as concisely as you can, using a single phrase or sentence if possible. If the question cannot be answered based on the information in the article, write \"unanswerable\". If the question is a yes/no question, answer \"yes\", \"no\", or \"unanswerable\". Do not provide any explanation.\n\nArticle: {context}\n\n Answer the question based on the above article as concisely as you can, using a single phrase or sentence if possible. If the question cannot be answered based on the information in the article, write \"unanswerable\". If the question is a yes/no question, answer \"yes\", \"no\", or \"unanswerable\". Do not provide any explanation.\n\nQuestion: {input}\n\nAnswer:",
    "multifieldqa_en": "Read the following text and answer briefly.\n\n{context}\n\nNow, answer the following question based on the above text, only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "2wikimqa": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "gov_report": "You are given a report by a government agency. Write a one-page summary of the report.\n\nReport:\n{context}\n\nNow, write a one-page summary of the report.\n\nSummary:",
    "multi_news": "You are given several news passages. Write a one-page summary of all news. \n\nNews:\n{context}\n\nNow, write a one-page summary of all the news.\n\nSummary:",
    "trec": "Please determine the type of the question below. Here are some examples of questions.\n\n{context}\n{input}",
    "triviaqa": "Answer the question based on the given passage. Only give me the answer and do not output any other words. The following are some examples.\n\n{context}\n\n{input}",
    "passage_retrieval_en": "Here are 30 paragraphs from Wikipedia, along with an abstract. Please determine which paragraph the abstract is from.\n\n{context}\n\nThe following is an abstract.\n\n{input}\n\nPlease enter the number of the paragraph that the abstract is from. The answer format must be like \"Paragraph 1\", \"Paragraph 2\", etc.\n\nThe answer is: ",
}

DATASET2MAXLEN = {
    "narrativeqa": 128,
    "qasper": 128,
    "multifieldqa_en": 64,
    "2wikimqa": 32,
    "gov_report": 512,
    "multi_news": 512,
    "trec": 64,
    "triviaqa": 32,
    "passage_retrieval_en": 32,
}

# These tasks do NOT get the chat template (from THUDM pred.py)
NO_CHAT_TEMPLATE = {"trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"}

EXPERIMENT_DEFINITIONS = {
    "baseline": {
        "injector": lambda args: {
            "delta_decode": False,
            "delta_pf_key_on": 0,
            "delta_mlp": "Regular",
            "flash": False,
        }
    },
    "decode_only_delta": {
        "grid": {
            "row_thresh": [13, 10],
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
# THUDM SCORING FUNCTIONS
# ==========================================

def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)
    def white_space_fix(text):
        return " ".join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)
    return white_space_fix(remove_articles(remove_punc(s.lower())))

def f1_score(prediction_tokens, ground_truth_tokens):
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(prediction_tokens)
    recall = num_same / len(ground_truth_tokens)
    return (2 * precision * recall) / (precision + recall)

def qa_f1_score(prediction, ground_truth, **kwargs):
    pred_tokens = normalize_answer(prediction).split()
    gt_tokens = normalize_answer(ground_truth).split()
    return f1_score(pred_tokens, gt_tokens)

from rouge import Rouge as RougeScorer

def rouge_l_score(prediction, ground_truth, **kwargs):
    rouge = RougeScorer()
    try:
        scores = rouge.get_scores([prediction], [ground_truth], avg=True)
    except:
        return 0.0
    return scores["rouge-l"]["f"]

def classification_score(prediction, ground_truth, **kwargs):
    all_classes = kwargs.get("all_classes", [])
    em_match_list = []
    for class_name in all_classes:
        if class_name in prediction:
            em_match_list.append(class_name)
    for match_term in list(em_match_list):
        if match_term in ground_truth and match_term != ground_truth:
            em_match_list.remove(match_term)
    if ground_truth in em_match_list:
        return 1.0 / len(em_match_list)
    return 0.0

def retrieval_score(prediction, ground_truth, **kwargs):
    pattern = r'Paragraph (\d+)'
    matches = re.findall(pattern, ground_truth)
    if not matches:
        return 0.0
    ground_truth_id = matches[0]
    numbers = re.findall(r"\d+", prediction)
    if not numbers:
        return 0.0
    right_num = sum(1 for n in numbers if n == ground_truth_id)
    return right_num / len(numbers)

DATASET2METRIC = {
    "narrativeqa": qa_f1_score,
    "qasper": qa_f1_score,
    "multifieldqa_en": qa_f1_score,
    "2wikimqa": qa_f1_score,
    "gov_report": rouge_l_score,
    "multi_news": rouge_l_score,
    "trec": classification_score,
    "triviaqa": qa_f1_score,
    "passage_retrieval_en": retrieval_score,
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
    file_path  = os.path.join(tasks_dir, f"longbench_{task_name}.json")
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


def build_prompt(json_obj, dataset, tokenizer, max_length):
    prompt_template = DATASET2PROMPT[dataset]
    prompt = prompt_template.format(**json_obj)

    # Middle truncation: keep first half + last half of token ids
    tokenized = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
    if len(tokenized) > max_length:
        half = max_length // 2
        prompt = (
            tokenizer.decode(tokenized[:half], skip_special_tokens=True)
            + tokenizer.decode(tokenized[-half:], skip_special_tokens=True)
        )

    # Apply Llama-2 chat template only for tasks that need it
    if dataset not in NO_CHAT_TEMPLATE:
        prompt = f"[INST]{prompt}[/INST]"

    return prompt


def score_prediction(prediction, ground_truths, all_classes, dataset):
    # For trec/triviaqa, THUDM only scores the first line
    if dataset in {"trec", "triviaqa", "samsum", "lsht"}:
        prediction = prediction.lstrip('\n').split('\n')[0]

    metric_fn = DATASET2METRIC[dataset]
    score = 0.0
    for gt in ground_truths:
        score = max(score, metric_fn(prediction, gt, all_classes=all_classes))
    return score


# ==========================================
# WORKER
# ==========================================

def run_worker_process(args, experiment_dir):
    torch.set_grad_enabled(False)

    exp_def       = EXPERIMENT_DEFINITIONS[args.experiment_type]
    glob_settings = exp_def["injector"](args)

    print(f"[Worker] Experiment settings: {glob_settings}", flush=True)

    # Safe defaults before model load
    setattr(globVR, 'delta_pf_key_on', 0)
    setattr(globVR, 'delta_mlp',      'Regular')
    setattr(globVR, 'delta_decode',   False)

    print(f"[Worker] Loading model: {MODEL_ID}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="sdpa",
    ).to("cuda").eval()

    # Inject experiment globals
    print(f"[Worker] Applying {args.experiment_type} settings...", flush=True)
    for key, value in glob_settings.items():
        setattr(globVR, key, value)

    # Patch generate() for delta decode
    if getattr(globVR, 'delta_decode', False):
        print("[Worker] Patching model.generate() for HybridCompressedCache...", flush=True)
        original_generate = model.generate

        def generate_with_custom_cache(*gen_args, **gen_kwargs):
            input_ids = gen_kwargs.get('input_ids', gen_args[0] if gen_args else None)
            bsz = input_ids.shape[0] if input_ids is not None else 1
            gen_kwargs['past_key_values'] = HybridCompressedCache(
                config=model.config,
                batch_size=bsz,
                dtype=model.dtype,
                exact_window_size=getattr(globVR, 'window_size', 100),
            )
            gen_kwargs['use_cache'] = True
            return original_generate(*gen_args, **gen_kwargs)

        model.generate = generate_with_custom_cache

    force_name = "config_baseline" if args.experiment_type == "baseline" else None
    config_dir  = get_or_create_config_dir(experiment_dir, glob_settings, force_dir_name=force_name)

    for dataset in TASKS:
        task_file = os.path.join(config_dir, "tasks", f"longbench_{dataset}.json")
        if os.path.exists(task_file):
            print(f"--- Skipping {dataset} (already done) ---", flush=True)
            continue

        print(f"--- Running Task: {dataset} ---", flush=True)

        data = load_dataset('THUDM/LongBench', dataset, split='test', revision='f72191f')
        max_new_tokens = DATASET2MAXLEN[dataset]

        scores = []
        seq_lengths = []
        task_start = time.time()

        for json_obj in data:
            prompt = build_prompt(json_obj, dataset, tokenizer, MAX_LENGTH)
            inputs = tokenizer(prompt, truncation=False, return_tensors="pt").to("cuda")
            context_length = inputs.input_ids.shape[-1]
            seq_lengths.append(context_length)

            with torch.no_grad():
                output = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    num_beams=1,
                    do_sample=False,
                    temperature=1.0,
                )[0]

            pred = tokenizer.decode(output[context_length:], skip_special_tokens=True)
            sample_score = score_prediction(
                pred,
                json_obj["answers"],
                json_obj["all_classes"],
                dataset,
            )
            scores.append(sample_score)

        total_time = time.time() - task_start
        accuracy = float(np.mean(scores))

        result_data = {
            "task_name": f"longbench_{dataset}",
            "timestamp": datetime.now().isoformat(),
            "shot": 0,
            "sparsity": 0.0,
            "mlp_spars": 0.0,
            "avg_kv_compression_pct": 0.0,
            "benchmark_stats": {
                "num_passages": len(seq_lengths),
                "seq_len_avg": float(np.mean(seq_lengths)),
                "seq_len_min": int(np.min(seq_lengths)),
                "seq_len_max": int(np.max(seq_lengths)),
                "seq_len_std": float(np.std(seq_lengths)),
            },
            "accuracy": accuracy,
            "total_benchmark_time_s": total_time,
            "parameters": glob_settings,
            "experiment_type": args.experiment_type,
            "model_id": MODEL_ID,
        }

        print(
            f"Task: {dataset} | Acc: {accuracy * 100:.2f} | Time: {total_time:.2f}s",
            flush=True,
        )
        save_single_task_result(config_dir, dataset, result_data)

        gc.collect()
        torch.cuda.empty_cache()

    print("[Worker] Finished.", flush=True)


# ==========================================
# MAIN
# ==========================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="LongBench eval — THUDM protocol, Llama-2-7B-chat, 4K context."
    )
    parser.add_argument('--mode', type=str, default='master', choices=['master', 'worker'])
    parser.add_argument(
        '--experiment_type', type=str, required=True,
        choices=list(EXPERIMENT_DEFINITIONS.keys()),
    )
    parser.add_argument('--exp_name', default=None, type=str)

    # Decode experiment args
    parser.add_argument('--row_thresh',  default=15,          type=float)
    parser.add_argument('--window_size', default=100,         type=int)
    parser.add_argument('--row_sim',     default="euclidean", type=str)

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
        keys       = list(grid_params.keys())   if grid_params else []
        values     = list(grid_params.values()) if grid_params else []
        param_grid = list(itertools.product(*values)) if grid_params else [{}]

        print(f"[Master] Experiment:  {args.experiment_type}", flush=True)
        print(f"[Master] Output:      {args.exp_name}",        flush=True)
        print(f"[Master] Tasks:       {TASKS}",                flush=True)

        for i, combo in enumerate(param_grid):
            current_params = dict(zip(keys, combo)) if keys else {}
            print(f"\n=== Step {i+1}/{len(param_grid)}: {current_params} ===", flush=True)

            cmd = [
                sys.executable, sys.argv[0],
                "--mode",            "worker",
                "--experiment_type", args.experiment_type,
                "--exp_name",        args.exp_name,
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
