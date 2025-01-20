import os
import subprocess

# Ensure the logs directory exists
os.makedirs('./logs', exist_ok=True)

commands = [
    #########################################################################################################################################
# # Table 1 clean parts
#     ["python", "./ar_evaluate_clean.py", "--model", "vicuna7b", "--end", "520"],
#     ["python", "./ar_evaluate_clean.py", "--model", "vicuna13b", "--end", "520"],
#     ["python", "./ar_evaluate_clean.py", "--model", "mistral", "--end", "520"],
#     ["python", "./ar_evaluate_clean.py", "--model", "llama7b", "--end", "520"],
#########################################################################################################################################

#########################################################################################################################################
# Table 1
    # ["python", "./run_attack_and_evaluate.py", "--end", "520", "--semantic_threshold", "0.005", "--model", "vicuna7b", "--budget", "15", "--attack_method", "beast"],
    # ["python", "./run_attack_and_evaluate.py", "--end", "520", "--semantic_threshold", "0.005", "--model", "vicuna13b", "--budget", "15", "--attack_method", "beast"],
    # ["python", "./run_attack_and_evaluate.py", "--end", "520", "--semantic_threshold", "0.005", "--model", "llama7b", "--budget", "15", "--attack_method", "beast"],
    # ["python", "./run_attack_and_evaluate.py", "--end", "520", "--semantic_threshold", "0.005", "--model", "mistral", "--budget", "15", "--attack_method", "beast"],


    # ["python", "./run_attack_and_evaluate.py", "--end", "520", "--semantic_threshold", "0.005", "--model", "vicuna7b", "--budget", "30", "--attack_method", "beast"],
    # ["python", "./run_attack_and_evaluate.py", "--end", "520", "--semantic_threshold", "0.005", "--model", "vicuna13b", "--budget", "30", "--attack_method", "beast"],
    # ["python", "./run_attack_and_evaluate.py", "--end", "520", "--semantic_threshold", "0.005", "--model", "llama7b", "--budget", "30", "--attack_method", "beast"],
    ["python", "./run_attack_and_evaluate.py", "--end", "520", "--semantic_threshold", "0.005", "--model", "mistral", "--budget", "30", "--attack_method", "beast"],


    ["python", "./run_attack_and_evaluate.py", "--end", "520", "--semantic_threshold", "0.005", "--model", "vicuna7b", "--budget", "60", "--attack_method", "beast"],
    ["python", "./run_attack_and_evaluate.py", "--end", "520", "--semantic_threshold", "0.005", "--model", "vicuna13b", "--budget", "60", "--attack_method", "beast"],
    # ["python", "./run_attack_and_evaluate.py", "--end", "520", "--semantic_threshold", "0.005", "--model", "llama7b", "--budget", "60", "--attack_method", "beast"],
    # ["python", "./run_attack_and_evaluate.py", "--end", "520", "--semantic_threshold", "0.005", "--model", "mistral", "--budget", "60", "--attack_method", "beast"],

    # ["python", "./run_attack_and_evaluate.py", "--end", "520", "--semantic_threshold", "0.001", "--model", "vicuna7b", "--budget", "15", "--attack_method", "beast"],
    # ["python", "./run_attack_and_evaluate.py", "--end", "520", "--semantic_threshold", "0.001", "--model", "vicuna7b", "--budget", "30", "--attack_method", "beast"],
    # ["python", "./run_attack_and_evaluate.py", "--end", "520", "--semantic_threshold", "0.001", "--model", "vicuna7b", "--budget", "60", "--attack_method", "beast"],

    # ["python", "./run_attack_and_evaluate.py", "--end", "520", "--semantic_threshold", "0.001", "--model", "vicuna13b", "--budget", "15", "--attack_method", "beast"],
    # ["python", "./run_attack_and_evaluate.py", "--end", "520", "--semantic_threshold", "0.001", "--model", "vicuna13b", "--budget", "30", "--attack_method", "beast"],
    # ["python", "./run_attack_and_evaluate.py", "--end", "520", "--semantic_threshold", "0.001", "--model", "vicuna13b", "--budget", "60", "--attack_method", "beast"],

    # ["python", "./run_attack_and_evaluate.py", "--end", "520", "--semantic_threshold", "0.001", "--model", "llama7b", "--budget", "15", "--attack_method", "beast"],
    # ["python", "./run_attack_and_evaluate.py", "--end", "520", "--semantic_threshold", "0.001", "--model", "llama7b", "--budget", "30", "--attack_method", "beast"],
    # ["python", "./run_attack_and_evaluate.py", "--end", "520", "--semantic_threshold", "0.001", "--model", "llama7b", "--budget", "60", "--attack_method", "beast"],

    # ["python", "./run_attack_and_evaluate.py", "--end", "520", "--semantic_threshold", "0.001", "--model", "mistral", "--budget", "15", "--attack_method", "beast"],
    # ["python", "./run_attack_and_evaluate.py", "--end", "520", "--semantic_threshold", "0.001", "--model", "mistral", "--budget", "30", "--attack_method", "beast"],
    # ["python", "./run_attack_and_evaluate.py", "--end", "520", "--semantic_threshold", "0.001", "--model", "mistral", "--budget", "60", "--attack_method", "beast"],


#########################################################################################################################################

#########################################################################################################################################
# Table 2
    # ["python", "./run_attack_and_evaluate.py", "--end", "130", "--semantic_threshold", "0.05", "--model", "vicuna7b", "--budget", "15", "--attack_method", "beast"],
    # ["python", "./run_attack_and_evaluate.py", "--end", "130", "--semantic_threshold", "0.01", "--model", "vicuna7b", "--budget", "15", "--attack_method", "beast"],
    # ["python", "./run_attack_and_evaluate.py", "--end", "130", "--semantic_threshold", "0.001", "--model", "vicuna7b", "--budget", "15", "--attack_method", "beast"],
    # ["python", "./run_attack_and_evaluate.py", "--end", "130", "--semantic_threshold", "0.001", "--model", "vicuna7b", "--budget", "15", "--attack_method", "beast"],

    # ["python", "./run_attack_and_evaluate.py", "--end", "130", "--semantic_threshold", "0.05", "--model", "vicuna7b", "--budget", "30", "--attack_method", "beast"],
    # ["python", "./run_attack_and_evaluate.py", "--end", "130", "--semantic_threshold", "0.01", "--model", "vicuna7b", "--budget", "30", "--attack_method", "beast"],
    # ["python", "./run_attack_and_evaluate.py", "--end", "130", "--semantic_threshold", "0.001", "--model", "vicuna7b", "--budget", "30", "--attack_method", "beast"],
    # ["python", "./run_attack_and_evaluate.py", "--end", "130", "--semantic_threshold", "0.001", "--model", "vicuna7b", "--budget", "30", "--attack_method", "beast"],

    # ["python", "./run_attack_and_evaluate.py", "--end", "130", "--semantic_threshold", "0.05", "--model", "vicuna7b", "--budget", "60", "--attack_method", "beast"],
    # ["python", "./run_attack_and_evaluate.py", "--end", "130", "--semantic_threshold", "0.01", "--model", "vicuna7b", "--budget", "60", "--attack_method", "beast"],
    # ["python", "./run_attack_and_evaluate.py", "--end", "130", "--semantic_threshold", "0.001", "--model", "vicuna7b", "--budget", "60", "--attack_method", "beast"],
    # ["python", "./run_attack_and_evaluate.py", "--end", "130", "--semantic_threshold", "0.001", "--model", "vicuna7b", "--budget", "60", "--attack_method", "beast"],

    # ["python", "./run_attack_and_evaluate.py", "--end", "130", "--semantic_threshold", "0.05", "--model", "vicuna7b", "--budget", "120", "--attack_method", "beast"],
    # ["python", "./run_attack_and_evaluate.py", "--end", "130", "--semantic_threshold", "0.01", "--model", "vicuna7b", "--budget", "120", "--attack_method", "beast"],
    # ["python", "./run_attack_and_evaluate.py", "--end", "130", "--semantic_threshold", "0.001", "--model", "vicuna7b", "--budget", "120", "--attack_method", "beast"],
    # ["python", "./run_attack_and_evaluate.py", "--end", "130", "--semantic_threshold", "0.001", "--model", "vicuna7b", "--budget", "120", "--attack_method", "beast"],

    # ["python", "./run_attack_and_evaluate.py", "--end", "130", "--semantic_threshold", "0.05", "--model", "vicuna7b", "--budget", "240", "--attack_method", "beast"],
    # ["python", "./run_attack_and_evaluate.py", "--end", "130", "--semantic_threshold", "0.01", "--model", "vicuna7b", "--budget", "240", "--attack_method", "beast"],
    # ["python", "./run_attack_and_evaluate.py", "--end", "130", "--semantic_threshold", "0.001", "--model", "vicuna7b", "--budget", "240", "--attack_method", "beast"],
    # ["python", "./run_attack_and_evaluate.py", "--end", "130", "--semantic_threshold", "0.001", "--model", "vicuna7b", "--budget", "240", "--attack_method", "beast"],
#########################################################################################################################################
]

log_file = open("./logs/run_multi_experimental_settings.log", "w")

for command in commands:
    print(f"Running command: {command}")
    subprocess.run(command, stdout=log_file, stderr=subprocess.STDOUT)

log_file.close()