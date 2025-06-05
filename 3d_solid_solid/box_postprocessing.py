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
# get paths for heat_box_outputs/fourier_net_128_silu/events.out.tfevents.* with name ml1.hpc.uio.no in it
fourier_net_128_silu_paths = []
for file in os.listdir(os.path.join(BASE_DIR, "heat_box_outputs", "fourier_net_128_silu")):
    if "ml1.hpc.uio.no" in file:
        fourier_net_128_silu_paths.append(os.path.join(BASE_DIR, "heat_box_outputs", "fourier_net_128_silu", file))
paths["fourier_net_128_silu"] = fourier_net_128_silu_paths
# get paths for heat_box_outputs/fourier_net_512_silu/events.out.tfevents.* with name ml1.hpc.uio.no in it
fourier_net_512_silu_paths = []
for file in os.listdir(os.path.join(BASE_DIR, "heat_box_outputs", "fourier_net_512_silu")):
    if "ml1.hpc.uio.no" in file:
        fourier_net_512_silu_paths.append(os.path.join(BASE_DIR, "heat_box_outputs", "fourier_net_512_silu", file))
paths["fourier_net_512_silu"] = fourier_net_512_silu_paths
# get paths for heat_box_outputs/fully_connected_128/events.out.tfevents.* with name ml1.hpc.uio.no in it
fully_connected_128_paths = []
for file in os.listdir(os.path.join(BASE_DIR, "heat_box_outputs", "fully_connected_128_silu")):
    if "ml1.hpc.uio.no" in file:
        fully_connected_128_paths.append(os.path.join(BASE_DIR, "heat_box_outputs", "fully_connected_128_silu", file))
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
plt.plot(data["fourier_net_128_silu"]["loss_aggregated"]["step"], data["fourier_net_128_silu"]["loss_aggregated"]["value"], label="Fourier Net 128")
plt.plot(data["fourier_net_512_silu"]["loss_aggregated"]["step"], data["fourier_net_512_silu"]["loss_aggregated"]["value"], label="Fourier Net 512")
plt.plot(data["fully_connected_128"]["loss_aggregated"]["step"], data["fully_connected_128"]["loss_aggregated"]["value"], label="Fully Connected 128")
plt.xlim(0, 300000)
plt.xlabel("Step")
plt.ylabel("Loss")
plt.legend(loc="upper left")
plt.grid()
plt.yscale("log")

ax_ins = inset_axes(ax, width="40%", height="40%", loc="upper right")

# repeat the same three curves inside the inset
for key, ls in [("fourier_net_128_silu", "-"),
                ("fourier_net_512_silu", "-"),
                ("fully_connected_128", "-")]:
    ax_ins.plot(data[key]["loss_aggregated"]["step"],
                data[key]["loss_aggregated"]["value"],
                ls)

# limit to zoomed region
ax_ins.set_xlim(220_000, 300_000)
ax_ins.set_ylim(0.35, 20)

ax_ins.set_yscale("log")
ax_ins.tick_params(axis="both", labelsize=8)
ax_ins.grid(True, which="both", linewidth=0.3)

# optional: draw lines linking inset ↔ full view
mark_inset(ax, ax_ins, loc1=2, loc2=4, fc="none", ec="0.4", lw=0.7)

plt.savefig(os.path.join(BASE_DIR, "heat_box_figures/box_loss_comparison.pdf"), dpi=300, bbox_inches="tight")

# plot l2_relative_error_theta
fig, ax = plt.subplots()
plt.plot(data["fourier_net_128_silu"]["l2_relative_error_theta"]["step"], data["fourier_net_128_silu"]["l2_relative_error_theta"]["value"], label="Fourier Net 128")
plt.plot(data["fourier_net_512_silu"]["l2_relative_error_theta"]["step"], data["fourier_net_512_silu"]["l2_relative_error_theta"]["value"], label="Fourier Net 512")
plt.plot(data["fully_connected_128"]["l2_relative_error_theta"]["step"], data["fully_connected_128"]["l2_relative_error_theta"]["value"], label="Fully Connected 128")
plt.xlabel("Step")
plt.xlim(0, 300000)
plt.ylabel(r"$l_2$ relative error")
plt.legend(loc="upper left")
plt.grid()
plt.yscale("log")

ax_ins = inset_axes(ax, width="40%", height="40%", loc="upper right")

# repeat the same three curves inside the inset
for key, ls in [("fourier_net_128_silu", "-"),
                ("fourier_net_512_silu", "-"),
                ("fully_connected_128", "-")]:
    ax_ins.plot(data[key]["l2_relative_error_theta"]["step"],
                data[key]["l2_relative_error_theta"]["value"],
                ls)

# limit to zoomed region
ax_ins.set_xlim(220_000, 300_000)
ax_ins.set_ylim(7.9e-2, 9.5e-2)

ax_ins.set_yscale("log")
ax_ins.tick_params(axis="both", labelsize=8)
ax_ins.grid(True, which="both", linewidth=0.3)

# optional: draw lines linking inset ↔ full view
mark_inset(ax, ax_ins, loc1=2, loc2=4, fc="none", ec="0.4", lw=0.7)

plt.savefig(os.path.join(BASE_DIR, "heat_box_figures/box_l2_relative_error_comparison.pdf"), dpi=300, bbox_inches="tight")

# plot grad_max
fig = plt.figure()
plt.plot(data["fourier_net_128_silu"]["grad_max"]["step"], data["fourier_net_128_silu"]["grad_max"]["value"], label="Fourier Net 128")
plt.plot(data["fourier_net_512_silu"]["grad_max"]["step"], data["fourier_net_512_silu"]["grad_max"]["value"], label="Fourier Net 512")
plt.plot(data["fully_connected_128"]["grad_max"]["step"], data["fully_connected_128"]["grad_max"]["value"], label="Fully Connected 128")

plt.xlabel("Step")
plt.xlim(0, 300000)
plt.ylabel("Max gradient")
plt.legend()
plt.grid()
plt.yscale("log")
plt.savefig(os.path.join(BASE_DIR, "heat_box_figures/box_grad_max_comparison.pdf"), dpi=300, bbox_inches="tight")
