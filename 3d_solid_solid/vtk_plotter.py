import pandas as pd
import tensorboard as tb
from tensorboard.backend.event_processing import event_accumulator
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib
from scipy.interpolate import griddata
from physicsnemo.sym.utils.io.plotter import _Plotter
from typing import List, Tuple
import vtk
from vtk.util.numpy_support import vtk_to_numpy

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

SLICE_NORMALS = {
    (0, 0, 1): "xy",
    (0, 1, 0): "xz",
    (1, 0, 0): "yz"
}

class VTKPlotter(_Plotter):
    def __init__(
            self,
            path_to_pinns: str,
            path_to_vtk: str,
            slice_origins: List[Tuple[float, float, float]] = (0, 0, -0.05),
            slice_normals: List[Tuple[float, float, float]] = (0, 0, 1),
            array_name="Temperature"
        ):
        """
        Initialize the VTKPlotter with paths to PINNs and VTK files.
        """
        self.path_to_pinns = path_to_pinns
        self.path_to_vtk = path_to_vtk
        self.slice_origins = slice_origins
        self.slice_normals = slice_normals
        self.array_name = array_name
    
    def __call__(self, *args):
        fs = self.plot_difference(self.path_to_pinns, self.path_to_vtk)
        return fs

    def slice_vtk_data(self, path_to_file, plane_origin, plane_normal, array_name="Temperature"):
        """
        Slice a VTK file using a plane defined by origin and normal.
        """
        reader = vtk.vtkXMLUnstructuredGridReader()
        reader.SetFileName(path_to_file)
        if reader.CanReadFile(path_to_file) == 0:
            raise ValueError(f"Cannot read VTK file: {path_to_file}")
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
    
    def plot_vtk_slice(self, pts, temp, slice_normal, label="Temperature", ax=None, **kwargs):
        """
        Plot a slice of VTK data.
        """
        if slice_normal == (0, 0, 1):
            x, y = pts[:, 0], pts[:, 1]
            x_label, y_label = "x, [mm]", "y, [mm]"
            kwargs["invert_yaxis"] = True
        elif slice_normal == (0, 1, 0):
            x, y = pts[:, 0], pts[:, 2]
            x_label, y_label = "x, [mm]", "z, [mm]"
        elif slice_normal == (1, 0, 0):
            x, y = pts[:, 1], pts[:, 2]
            x_label, y_label = "y, [mm]", "z, [mm]"
        else:
            raise ValueError("Unsupported slice normal. Use (0, 0, 1), (0, 1, 0), or (1, 0, 0).")
        xs = [x, y]
        extent = (xs[0].min(), xs[0].max(), xs[1].min(), xs[1].max())
        xyi = np.meshgrid(
            np.linspace(extent[0], extent[1], 300),
            np.linspace(extent[2], extent[3], 300),
            indexing="ij",
        )
        interpolated = griddata(
            (xs[0], xs[1]), temp, tuple(xyi), method="cubic"
        )
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 5), dpi=100)
        if "invert_yaxis" in kwargs and kwargs["invert_yaxis"]:
            ax.invert_yaxis()
        im = ax.imshow(
            interpolated.T,
            extent=extent,
            cmap="coolwarm",
            vmin=np.min(temp),
            vmax=np.max(temp),
        )
        ax.set_title(label)
        ax.set_aspect("equal")
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        plt.colorbar(im, ax=ax, label="Temperature, [C]")
        return ax

    def plot_difference(self, pinns_path, fem_path):
        """
        Plot the difference between PINNs and FEM solutions.
        """
        plots = []
        for slice_origin, slice_normal in zip(self.slice_origins, self.slice_normals):
            pinns_pts, pinns_conn, pinns_temp = self.slice_vtk_data(pinns_path, slice_origin, slice_normal, array_name=self.array_name)
            fem_pts, fem_conn, fem_temp = self.slice_vtk_data(fem_path, slice_origin, slice_normal, array_name="Temperature")

            fig, axes = plt.subplots(
                nrows=1, ncols=3, figsize=(16, 4), dpi=300, constrained_layout=True
            )

            pinns_plot = self.plot_vtk_slice(pinns_pts, pinns_temp, slice_normal, label="PINNs", ax=axes[0])
            fem_plot = self.plot_vtk_slice(fem_pts, fem_temp, slice_normal, label="FEM", ax=axes[1])
            diff_plot = self.plot_vtk_slice(pinns_pts, fem_temp - pinns_temp, slice_normal, label="Difference", ax=axes[2])
            plots.append((fig, SLICE_NORMALS[slice_normal]))
        return plots

if __name__ == "__main__":
    # Example usage
    plotter = VTKPlotter(
        path_to_pinns=os.path.join(BASE_DIR, "outputs/fixed/fourier_net_128_silu/inferencers/vtk_inf.vtu"),
        path_to_vtk=os.path.join(BASE_DIR, "temp_sol_5_fins.vtu"),
        slice_origins=[(0, 0, -0.02449326630430), (0, 0, 0)],
        slice_normals=[(0, 0, 1), (0, 1, 0)],
        array_name="Temperature"
    )
    plotter()
    plt.show()  # Show the plots