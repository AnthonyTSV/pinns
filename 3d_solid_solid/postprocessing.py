import pandas as pd
import tensorboard as tb
from tensorboard.backend.event_processing import event_accumulator
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib
import scienceplots
from scipy.interpolate import griddata
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

plt.style.use("science")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

matplotlib.rcParams["figure.figsize"] = (8, 6)
matplotlib.rcParams["font.size"] = 16

def load_event(path):
    ea = event_accumulator.EventAccumulator(path, size_guidance={event_accumulator.TENSORS: 0})
    ea.Reload()
    return ea

def get_data(ea):
    data = {}
    for tensor in ea.Tags()["tensors"]:
        if tensor == 'config/text_summary':
            continue
        tensor_name = tensor.split("/")[-1]
        data[tensor_name] = {"step": [], "value": []}
        tensor_data = ea.Tensors(tensor)
        for tensor_event in tensor_data:
            step = tensor_event.step
            value = tensor_event.tensor_proto.float_val
            if isinstance(value, list):
                value = value[0]
            data[tensor_name]["step"].append(step)
            data[tensor_name]["value"].append(value)
        data[tensor_name]["step"] = np.array(data[tensor_name]["step"])
        data[tensor_name]["value"] = np.array(data[tensor_name]["value"])
    return data

def get_combined_data(ea_list):
    data = {}
    for ea in ea_list:
        ea_data = get_data(ea)
        for key in ea_data:
            if key not in data:
                data[key] = {"step": [], "value": []}
            data[key]["step"].extend(ea_data[key]["step"])
            data[key]["value"].extend(ea_data[key]["value"])
    for key in data:
        data[key]["step"] = np.array(data[key]["step"])
        data[key]["value"] = np.array(data[key]["value"])
    return data

paths = {
    "fourier_net_128_silu": [],
    "fourier_net_512_silu": [],
    "fully_connected_128": [],
}

def get_paths_for_model(model_name):
    model_paths = []
    for file in os.listdir(os.path.join(BASE_DIR, "outputs/fixed", model_name)):
        model_paths.append(os.path.join(BASE_DIR, "outputs/fixed", model_name, file))
    return model_paths

fully_connected_128_paths = get_paths_for_model("fully_connected_128_silu")
paths["fully_connected_128"] = fully_connected_128_paths
data = {}
for key, path in paths.items():
    if isinstance(path, list):
        ea_list = [load_event(p) for p in path]
        data[key] = get_combined_data(ea_list)
    else:
        ea = load_event(path)
        data[key] = get_data(ea)

# plot loss_aggregated
# limit the x-axis to 300000 steps
fig, ax = plt.subplots()
plt.plot(data["fully_connected_128"]["loss_aggregated"]["step"], data["fully_connected_128"]["loss_aggregated"]["value"], label="Fully Connected 128")
plt.xlabel("Step")
plt.ylabel("Loss")
plt.legend(loc="upper left")
plt.grid()
plt.yscale("log")
plt.show()