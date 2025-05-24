import yaml
import os
import sys
import subprocess
from heat_2d import run

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def read_and_set_config(activation_func, batch_size, architecture):
    # read config.yaml file and set the parameters
    with open(BASE_DIR+"/conf/config.yaml", "r") as f:
        config = yaml.safe_load(f)
        config["custom"]["activation"] = activation_func
        config["batch_size"]["interior"] = batch_size
        config["custom"]["layer_size"] = architecture[0]
        config["custom"]["num_layers"] = architecture[1]
    
    # write the modified config back to the file
    with open(BASE_DIR+"/conf/config.yaml", "w") as f:
        yaml.dump(config, f, indent=4, sort_keys=False)

def run_experiment_w_act_funcs(activation_funcs, batch_size, architecture):
    for activation_func in activation_funcs:
        print(f"Running with activation function: {activation_func}")
        read_and_set_config(activation_func, batch_size, architecture)
        subprocess.run([sys.executable, BASE_DIR+"/heat_2d.py"])

def run_experiment_w_batch_sizes(batch_sizes, activation_func, architecture):
    for batch_size in batch_sizes:
        print(f"Running with batch size: {batch_size}")
        read_and_set_config(activation_func, batch_size, architecture)
        subprocess.run([sys.executable, BASE_DIR+"/heat_2d.py"])

def run_experiment_w_architectures(architectures, activation_func, batch_size):
    for architecture in architectures:
        print(f"Running with architecture: {architecture}")
        read_and_set_config(activation_func, batch_size, architecture)
        subprocess.run([sys.executable, BASE_DIR+"/heat_2d.py"])

if __name__ == "__main__":
    activation_funcs = ["tanh", "relu", "sigmoid", "silu"]
    batch_sizes = [2000, 4000]
    achitectures = [(64, 1), (128, 2), (128, 3), (256, 4), (256, 5), (256, 6)]
    # read config.yaml file and set the parameters

    run_experiment_w_act_funcs(activation_funcs, 2000, (128, 2))
    run_experiment_w_batch_sizes(batch_sizes, "silu", (128, 2))
    run_experiment_w_architectures(achitectures, "silu", 2000)