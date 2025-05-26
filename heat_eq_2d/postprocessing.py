import pandas as pd
import tensorboard as tb
from tensorboard.backend.event_processing import event_accumulator
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib
import scienceplots
plt.style.use("science")

matplotlib.rcParams["figure.figsize"] = (8, 6)
matplotlib.rcParams["font.size"] = 16

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

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

activation_funcs = ["tanh", "relu", "sigmoid", "silu"]
batch_sizes = [2000, 4000]
achitectures = [(64, 1), (128, 2), (128, 3), (256, 4), (256, 5), (256, 6)]

# plot experiment with activation functions
# activation_funcs, 2000, (128, 3)

paths = {
    "tanh": BASE_DIR + "/outputs/tanh_interior_2000_arch_128x3/events.out.tfevents.1748179220.ml1.hpc.uio.no.3493322.0",
    "relu": BASE_DIR + "/outputs/relu_interior_2000_arch_128x3/events.out.tfevents.1748193178.ml1.hpc.uio.no.3521357.0",
    "sigmoid": BASE_DIR + "/outputs/sigmoid_interior_2000_arch_128x3/events.out.tfevents.1748194029.ml1.hpc.uio.no.3522996.0",
    "silu": BASE_DIR + "/outputs/silu_interior_2000_arch_128x3/events.out.tfevents.1748194961.ml1.hpc.uio.no.3524950.0",
    "leaky_relu": BASE_DIR + "/outputs/leaky_relu_interior_2000_arch_128x3/events.out.tfevents.1748195906.ml1.hpc.uio.no.3526702.0"
}

data = {}
for func, path in paths.items():
    if isinstance(path, list):
        ea_list = [load_event(p) for p in path]
        data[func] = get_combined_data(ea_list)
    else:
        ea = load_event(path)
        data[func] = get_data(ea)

# plot loss_aggregated
fig = plt.figure()
plt.plot(data["tanh"]["loss_aggregated"]["step"], data["tanh"]["loss_aggregated"]["value"], label="Tanh")
plt.plot(data["relu"]["loss_aggregated"]["step"], data["relu"]["loss_aggregated"]["value"], label="ReLU")
plt.plot(data["sigmoid"]["loss_aggregated"]["step"], data["sigmoid"]["loss_aggregated"]["value"], label="Sigmoid")
plt.plot(data["silu"]["loss_aggregated"]["step"], data["silu"]["loss_aggregated"]["value"], label="SiLU")
plt.plot(data["leaky_relu"]["loss_aggregated"]["step"], data["leaky_relu"]["loss_aggregated"]["value"], label="Leaky ReLU")
plt.xlabel("Step")
plt.ylabel("Loss")
plt.grid()
plt.yscale("log")
plt.legend()
plt.savefig(BASE_DIR + "/figures/losses_activation_functions.pdf", dpi=300)

# plot l2_relative_error_theta
fig = plt.figure()
plt.plot(data["tanh"]["l2_relative_error_theta"]["step"], data["tanh"]["l2_relative_error_theta"]["value"], label="Tanh")
plt.plot(data["relu"]["l2_relative_error_theta"]["step"], data["relu"]["l2_relative_error_theta"]["value"], label="ReLU")
plt.plot(data["sigmoid"]["l2_relative_error_theta"]["step"], data["sigmoid"]["l2_relative_error_theta"]["value"], label="Sigmoid")
plt.plot(data["silu"]["l2_relative_error_theta"]["step"], data["silu"]["l2_relative_error_theta"]["value"], label="SiLU")
plt.plot(data["leaky_relu"]["l2_relative_error_theta"]["step"], data["leaky_relu"]["l2_relative_error_theta"]["value"], label="Leaky ReLU")
plt.xlabel("Step")
plt.ylabel(r"$l_2$ relative error")
plt.grid()
plt.yscale("log")
plt.legend()
plt.savefig(BASE_DIR + "/figures/l2_relative_error_theta_activation_functions.pdf", dpi=300)

# get data for batch sizes
# batch_sizes, "tanh", (128, 2)

paths = {
    2000: BASE_DIR + "/outputs/tanh_interior_2000_arch_256x4/events.out.tfevents.1748196757.ml1.hpc.uio.no.3528539.0",
    4000: BASE_DIR + "/outputs/tanh_interior_4000_arch_256x4/events.out.tfevents.1748197880.ml1.hpc.uio.no.3530858.0"
}

data = {}
for batch_size, path in paths.items():
    ea = load_event(path)
    data[batch_size] = get_data(ea)

# plot loss_aggregated
fig = plt.figure()
plt.plot(data[2000]["loss_aggregated"]["step"], data[2000]["loss_aggregated"]["value"], label="Batch size 2000")
plt.plot(data[4000]["loss_aggregated"]["step"], data[4000]["loss_aggregated"]["value"], label="Batch size 4000")
plt.xlabel("Step")
plt.ylabel("Loss")
plt.grid()
plt.yscale("log")
plt.legend()
plt.savefig(BASE_DIR + "/figures/losses_batch_sizes.pdf", dpi=300)
# plot l2_relative_error_theta
fig = plt.figure()
plt.plot(data[2000]["l2_relative_error_theta"]["step"], data[2000]["l2_relative_error_theta"]["value"], label="Batch size 2000")
plt.plot(data[4000]["l2_relative_error_theta"]["step"], data[4000]["l2_relative_error_theta"]["value"], label="Batch size 4000")
plt.xlabel("Step")
plt.ylabel(r"$l_2$ relative error")
plt.grid()
plt.yscale("log")
plt.legend()
plt.savefig(BASE_DIR + "/figures/l2_relative_error_theta_batch_sizes.pdf", dpi=300)

# get data for architectures
# achitectures, "tanh", 2000

paths = {
    "64x1": BASE_DIR + "/outputs/tanh_interior_2000_arch_64x1/events.out.tfevents.1748199261.ml1.hpc.uio.no.3533312.0",
    "128x2": BASE_DIR + "/outputs/tanh_interior_2000_arch_128x2/events.out.tfevents.1748200049.ml1.hpc.uio.no.3535027.0",
    "128x3": BASE_DIR + "/outputs/tanh_interior_2000_arch_128x3/events.out.tfevents.1748179220.ml1.hpc.uio.no.3493322.0",
    "256x4": BASE_DIR + "/outputs/tanh_interior_2000_arch_256x4/events.out.tfevents.1748196757.ml1.hpc.uio.no.3528539.0",
    "256x5": BASE_DIR + "/outputs/tanh_interior_2000_arch_256x5/events.out.tfevents.1748200982.ml1.hpc.uio.no.3537179.0",
    "256x6": BASE_DIR + "/outputs/tanh_interior_2000_arch_256x6/events.out.tfevents.1748202132.ml1.hpc.uio.no.3539885.0"
}

data = {}
for arch, path in paths.items():
    ea = load_event(path)
    data[arch] = get_data(ea)
# plot loss_aggregated
fig = plt.figure()
plt.plot(data["64x1"]["loss_aggregated"]["step"], data["64x1"]["loss_aggregated"]["value"], label="64x1")
plt.plot(data["128x2"]["loss_aggregated"]["step"], data["128x2"]["loss_aggregated"]["value"], label="128x2")
plt.plot(data["128x3"]["loss_aggregated"]["step"], data["128x3"]["loss_aggregated"]["value"], label="128x3")
plt.plot(data["256x4"]["loss_aggregated"]["step"], data["256x4"]["loss_aggregated"]["value"], label="256x4")
plt.plot(data["256x5"]["loss_aggregated"]["step"], data["256x5"]["loss_aggregated"]["value"], label="256x5")
plt.plot(data["256x6"]["loss_aggregated"]["step"], data["256x6"]["loss_aggregated"]["value"], label="256x6")
plt.xlabel("Step")
plt.ylabel("Loss")
plt.grid()
plt.yscale("log")
plt.legend()
plt.savefig(BASE_DIR + "/figures/losses_architectures.pdf", dpi=300)
# plot l2_relative_error_theta
fig = plt.figure()
plt.plot(data["64x1"]["l2_relative_error_theta"]["step"], data["64x1"]["l2_relative_error_theta"]["value"], label="64 x 1")
plt.plot(data["128x2"]["l2_relative_error_theta"]["step"], data["128x2"]["l2_relative_error_theta"]["value"], label="128 x 2")
plt.plot(data["128x3"]["l2_relative_error_theta"]["step"], data["128x3"]["l2_relative_error_theta"]["value"], label="128 x 3")
plt.plot(data["256x4"]["l2_relative_error_theta"]["step"], data["256x4"]["l2_relative_error_theta"]["value"], label="256 x 4")
plt.plot(data["256x5"]["l2_relative_error_theta"]["step"], data["256x5"]["l2_relative_error_theta"]["value"], label="256 x 5")
plt.plot(data["256x6"]["l2_relative_error_theta"]["step"], data["256x6"]["l2_relative_error_theta"]["value"], label="256 x 6")
plt.xlabel("Step")
plt.ylabel(r"$l_2$ relative error")
plt.grid()
plt.yscale("log")
plt.legend()
plt.savefig(BASE_DIR + "/figures/l2_relative_error_theta_architectures.pdf", dpi=300)