import globVR
import glob_set
import argparse
from lm_eval.evaluator import simple_evaluate
import os
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--shot', default=0, type=int)
args = parser.parse_args()
if args.shot == 0:
    tasks = ["arc_easy", "arc_challenge", "openbookqa", "boolq", "hellaswag", "piqa", "winogrande"]
    # tasks = ["arc_challenge"]
if args.shot == 5:
    tasks = ["triviaqa", "mmlu"]
if args.shot == 10:
    tasks = ["commonsense_qa", "truthfulqa_mc2"]

# tasks = ["squadv2"]
# tasks = ["gsm8k"]
tasks = ["arc_challenge"]

# model_args = "pretrained=/projects/0/prjs1280/huggingface_cache/llama3_2_1B-local/models--meta-llama--Llama-3.2-1B-Instruct/snapshots/9213176726f574b556790deb65791e0c5aa438b6,trust_remote_code=False,dtype=bfloat16,device_map=auto,attn_implementation=eager"
model_args = "pretrained=microsoft/bitnet-b1.58-2B-4T"

sparsity = {}
eval_result = []
# long_count = {}
n_sample = 100
for t in tasks:
    task = [t]
    results = simple_evaluate(
        model="hf",
        model_args=model_args,
        tasks=task,
        device="cuda:0",
        batch_size=1,
        num_fewshot=args.shot,
        limit = n_sample
    )
    sparsity[t] = globVR.spars
    # long_count[t] = globVR.count
    print(f'{task} delta matrix sparsity: {globVR.spars}')
    globVR.spars = 0.0
    # count = 0
    eval_result.append(results)


    save_dir = "./BitNet/checkpoints"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"bitnet_arcc_t1.pt")
    print(globVR.delta_key[0].shape)
    torch.save(globVR.delta_key, save_path)

for r in eval_result:
    for task_name, metrics in r["results"].items():
        print(f"=== {task_name} ===")
        for metric_name, value in metrics.items():
            print(f"{metric_name:15s} {value}")
        print()

for task_name, spar in sparsity.items():
    print(f"{task_name} sparsity: {spar}")

    
# for task_name, ct in long_count.items():
#     print(f"{task_name} number of samples longer than blk size: {ct}")

# if args.shot == 0:
#     for t, metrics in results["results"].items():
#         print(t)
#         acc = metrics["acc,none"]
#         acc_norm = metrics["acc_norm,none"]
#         print(f"{t:15s} 0-shot accuracy: {acc:.4f}, length normalized accuracy: {acc_norm:.4f}")
# elif args.shot == 5:
#     print(f"TriviaQA Metrices: {results["results"]["triviaqa"].keys()}")
#     print(f"MMLU Metrices: {results["results"]["mmlu"].keys()}")

# print("BoolQ accuracy:",      results["results"]["boolq"]["acc,none"])
# print("HellaSwag accuracy:",  results["results"]["hellaswag"]["acc,none"])
# print("PIQA accuracy:     ", results["results"]["piqa"]["acc,none"])
# print("Winogrande accuracy:", results["results"]["winogrande"]["acc,none"])