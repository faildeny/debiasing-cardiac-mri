import argparse
import os
import socket
from datetime import datetime

parser = argparse.ArgumentParser(description="Run a job")
parser.add_argument("gpu_id", type=int, help="GPU ID to use")

args = parser.parse_args()
gpu_id = args.gpu_id
current_time = datetime.now().strftime("%b%d_%H-%M-%S")
log_dir = os.path.join("logs", current_time + "_" + socket.gethostname())
os.makedirs(log_dir, exist_ok=True)
command = (
    f"nohup python cardioai/training.py --gpu_id {gpu_id} --experiment_dir {log_dir}"
)

command += f" > {log_dir}/log.txt 2>&1 &"
command_file = open(f"{log_dir}/command.txt", "w")
n = command_file.write(command)
command_file.close()
print(command)

os.system(command)
