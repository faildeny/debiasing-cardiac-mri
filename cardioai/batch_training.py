import argparse
import os
import yaml
import socket
import subprocess
from datetime import datetime
import shutil

config_file = "config.yaml"
with open(config_file) as file:
    config = yaml.safe_load(file)

parser = argparse.ArgumentParser(description="Run a job")
parser.add_argument("gpu_id", type=int, help="GPU ID to use")
# Select type of training kfold or repeated with different seeds
parser.add_argument("--kfold", action="store_true", help="Run kfold training")
parser.add_argument(
    "--repeated", action="store_true", help="Run repeated training with different seeds on same fold"
)
args = parser.parse_args()
gpu_id = args.gpu_id
kfold = args.kfold
repeated = args.repeated
if kfold and repeated:
    raise ValueError("Cannot set both kfold and repeated to True")

current_time = datetime.now().strftime("%b%d_%H-%M-%S")
appendix = config["experiment_name"]
base_dir = os.path.join("logs", current_time + "_" + socket.gethostname() + "_" + appendix)
os.makedirs(base_dir, exist_ok=True)
shutil.copy(config_file, base_dir)


def run_training(log_dir, gpu_id, fold, seed):
    os.makedirs(log_dir, exist_ok=True)
    shutil.copy(base_dir + "/" + config_file, log_dir)

    command = f"python cardioai/training.py --gpu_id {gpu_id} --experiment_dir {log_dir} --fold {fold}"

    command += f" > {log_dir}/log.txt 2>&1"
    command_file = open(f"{log_dir}/command.txt", "w")
    n = command_file.write(command)
    command_file.close()
    print(command)
    # command = "sleep 10"
    training_process = subprocess.run(command, shell=True)


if kfold:
    print("Running kfold training")
    for fold in range(0, config["params"]["n_splits"]):
        log_dir = os.path.join(base_dir, f"fold_{fold}")
        run_training(log_dir, gpu_id, fold, 0)
    print("KFold training and evaluation finished")

if repeated:
    print("Running repeated training")
    fold = 0
    for seed in range(0, config["params"]["n_repeats"]):
        log_dir = os.path.join(base_dir, f"seed_{seed}")
        run_training(log_dir, gpu_id, fold, seed)
    print("Repeated training and evaluation finished")

command = (
    f"python cardioai/compile_results.py {base_dir} > {base_dir}/log_compile.txt 2>&1"
)
subprocess.run(command, shell=True)

print("Batch training and evaluation finished")
