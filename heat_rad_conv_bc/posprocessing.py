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
path = BASE_DIR + "/outputs/silu_interior_4000_arch_256x5/events.out.tfevents.1747909900.Oppenheimer.17914.0"
ea = event_accumulator.EventAccumulator(path, size_guidance={event_accumulator.TENSORS: 0})

ea.Reload()
print(ea.Tags())

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

# plot l2_relative_error_theta_phys
l2_relative_error_theta_phys = data["l2_relative_error_theta_phys"]

fig = plt.figure()
plt.plot(l2_relative_error_theta_phys["step"], l2_relative_error_theta_phys["value"])
plt.xlabel("Step")
plt.ylabel(r"$l_2$ relative error")
plt.grid()
plt.savefig(BASE_DIR + "/figures/l2_relative_error_theta_phys.pdf", dpi=300)

# plot loss_diffusion_theta, loss_convective_theta, loss_normal_gradient_theta and loss_aggregated

fig = plt.figure()
plt.plot(data["loss_diffusion_theta"]["step"], data["loss_diffusion_theta"]["value"], label="Diffusion")
plt.plot(data["loss_convective_theta"]["step"], data["loss_convective_theta"]["value"], label="Convective")
plt.plot(data["loss_normal_gradient_theta"]["step"], data["loss_normal_gradient_theta"]["value"], label="Insulated")
plt.plot(data["loss_aggregated"]["step"], data["loss_aggregated"]["value"], label="Aggregated")
plt.xlabel("Step")
plt.ylabel("Loss")
plt.grid()
plt.yscale("log")
plt.legend()
plt.savefig(BASE_DIR + "/figures/losses.pdf", dpi=300)