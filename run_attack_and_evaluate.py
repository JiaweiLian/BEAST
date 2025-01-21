import subprocess
import os
import sys
import argparse
import time

start_time = time.time()

parser = argparse.ArgumentParser()
parser.add_argument("--length", type=int, help="Length of the generated adversarial prompt", default=40)
parser.add_argument("--end", type=int, help="End argument", default=100)
parser.add_argument("--semantic_threshold", type=float, help="Semantic threshold argument", default=0.01)
parser.add_argument("--model", type=str, help="Model argument", default="vicuna7b")
# parser.add_argument("--GPU_index", type=int, help="GPU index", default=1)
parser.add_argument("--budget", type=int, help="Per sample attack budget in seconds", default=60)
parser.add_argument("--attack_method", type=str, help="Attack method", default="abs")
parser.add_argument('--DIR', type=str, default='logs/run/')
parser.add_argument('--clean', type=str, default=0)
args = parser.parse_args()

if args.attack_method == 'beast':
    args.semantic_threshold = 0.0
   
commands = [
    # attack
    ["python", "ar_self_attack.py", "--k1", "15", "--k2", "15", "--length", f"{args.length}", "--model", f"{args.model}", "--log", "1", "--target", "1", "--begin", "0", "--end", f"{args.end}", "--semantic_threshold", f"{args.semantic_threshold}", "--budget", f"{args.budget}", "--attack_method", f"{args.attack_method}"],
    # evaluate
    # ["python", "ar_evaluate.py", "--model", f"{args.model}", "--file_name", f"./logs/adv_tokens/{args.model}_end{args.end}_threshold={args.semantic_threshold}_budget={args.budget}_{args.attack_method}.pkl"],
]

if not os.path.exists(args.DIR):
    os.makedirs(args.DIR)
    print(f"Directory '{args.DIR}' created.")
else:
    print(f"Directory '{args.DIR}' already exists.")

log_file = open(os.path.join(args.DIR, f"{args.model}_end{args.end}_threshold{args.semantic_threshold}_budget{args.budget}_{args.attack_method}.log"), "w")

for command in commands:
    env = os.environ.copy()
    # env["CUDA_VISIBLE_DEVICES"] = f"{args.GPU_index}"  # Specify the GPU index here
    subprocess.run(command, stdout=log_file, stderr=subprocess.STDOUT, env=env)

log_file.close()

end_time = time.time()
execution_time = end_time - start_time
hours = int(execution_time // 3600)
minutes = int((execution_time % 3600) // 60)
seconds = int(execution_time % 60)

print(f"Execution time: {hours} hours, {minutes} minutes, {seconds} seconds")