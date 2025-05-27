import pandas as pd
import tensorboard as tb
from tensorboard.backend.event_processing import event_accumulator
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib
import scienceplots
from scipy.interpolate import griddata
import vtk
from vtk.util.numpy_support import vtk_to_numpy

plt.style.use("science")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def slice_vtk_data(path_to_file, plane_origin, plane_normal, array_name="Temperature"):
    """
    Slice a VTK file using a plane defined by origin and normal.
    """
    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(path_to_file)
    reader.Update()

    ugrid = reader.GetOutput()
    
    plane = vtk.vtkPlane()
    plane.SetOrigin(plane_origin)
    plane.SetNormal(plane_normal)

    cutter = vtk.vtkCutter()
    cutter.SetInputData(ugrid)
    cutter.SetCutFunction(plane)
    cutter.Update()

    slice_poly = cutter.GetOutput()

    points_vtk = slice_poly.GetPoints().GetData()
    tris_vtk   = slice_poly.GetPolys().GetData()
    temp_vtk   = slice_poly.GetPointData().GetArray(array_name)

    pts  = vtk_to_numpy(points_vtk)
    conn = vtk_to_numpy(tris_vtk).reshape(-1, 4)[:, 1:4]
    temp = vtk_to_numpy(temp_vtk)

    return pts, conn, temp

def plot_vtk_slice(path_to_file, plane_origin, plane_normal):
    """
    Plot a slice of VTK data.
    """
    pts, conn, temp = slice_vtk_data(path_to_file, plane_origin, plane_normal)

    x, y = pts[:, 0], pts[:, 1]
    xs = [x, y]
    extent = (xs[0].min(), xs[0].max(), xs[1].min(), xs[1].max())
    xyi = np.meshgrid(
        np.linspace(extent[0], extent[1], 100),
        np.linspace(extent[2], extent[3], 100),
        indexing="ij",
    )
    interpolated = griddata(
        (xs[0], xs[1]), temp, tuple(xyi)
    )

    fig, ax = plt.subplots(figsize=(6, 5), dpi=100)
    ax.imshow(
        interpolated.T, origin="lower", extent=extent, cmap="coolwarm", vmin=min(temp), vmax=max(temp)
    )
    ax.set_aspect("equal")
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    plt.tight_layout()
    plt.show()

def plot_difference(pinns_path, fem_path):
    """
    Plot the difference between PINNs and FEM solutions.
    """
    pinns_pts, pinns_conn, pinns_temp = slice_vtk_data(pinns_path, [-0.65, -0.4, -0.4375], (0, 0, 1), array_name="Temperature_true")
    fem_pts, fem_conn, fem_temp = slice_vtk_data(fem_path, [-0.65, -0.4, -0.4375], (0, 0, 1), array_name="Temperature")

    diff_temp = fem_temp - pinns_temp
    x, y = pinns_pts[:, 0], pinns_pts[:, 1]
    xs = [x, y]
    extent = (xs[0].min(), xs[0].max(), xs[1].min(), xs[1].max())
    xyi = np.meshgrid(
        np.linspace(extent[0], extent[1], 100),
        np.linspace(extent[2], extent[3], 100),
        indexing="ij",
    )
    interpolated_diff = griddata(
        (xs[0], xs[1]), diff_temp, tuple(xyi)
    )
    fig, ax = plt.subplots(figsize=(6, 5), dpi=100)
    plot = ax.imshow(
        interpolated_diff.T, origin="lower", extent=extent, cmap="coolwarm", vmin=min(diff_temp), vmax=max(diff_temp)
    )
    cbar = plt.colorbar(plot, ax=ax)
    cbar.set_label("Temperature Difference [C]")
    ax.set_aspect("equal")
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    plt.tight_layout()
    plt.show()

pinns_path = BASE_DIR + "/outputs/fully_connected/inferencers/vtk_inf.vtu"
fem_path = BASE_DIR + "/temp_sol.vtu"

plot_difference(pinns_path, fem_path)


