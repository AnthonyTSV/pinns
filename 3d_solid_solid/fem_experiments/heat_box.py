import numpy as np
import os
import ufl
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import fem, mesh, io, plot, default_scalar_type
from dolfinx.fem.petsc import LinearProblem, NonlinearProblem
from dolfinx.nls.petsc import NewtonSolver
from dolfinx.io import XDMFFile, VTKFile, gmshio
from dolfinx.fem.petsc import (
    assemble_matrix,
    assemble_vector,
    create_vector,
    apply_lifting,
    set_bc,
)
from typing import List, Tuple
import pyvista as pv

from mpi4py import MPI
import gmsh
from dolfinx.io import gmshio
from dolfinx.mesh import meshtags, locate_entities_boundary, locate_entities
import ufl.constant
# from .generate_mesh import main

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
comm = MPI.COMM_WORLD

x0, y0, z0 = -1, -1, -1
dx_, dy_, dz_ = 2, 2, 2
domain_mesh = mesh.create_box(
    comm,
    [np.array([-dx_/2, -dy_/2, -dz_/2]), np.array([dx_/2, dy_/2, dz_/2])],
    [10, 10, 10]
)

fdim = domain_mesh.topology.dim - 1

V = fem.functionspace(domain_mesh, ("Lagrange", 2))

on_bottom = lambda x: np.isclose(x[2], -dz_/2)
not_on_bottom = lambda x: np.logical_not(on_bottom(x))
bottom_facets = locate_entities_boundary(domain_mesh, fdim, on_bottom)
not_bottom_facets = locate_entities_boundary(domain_mesh, fdim, not_on_bottom)
num_facets = domain_mesh.topology.index_map(fdim).size_local
markers = np.zeros(num_facets, dtype=np.int64)
markers[bottom_facets] = 1
markers[not_bottom_facets] = 2
T = fem.Function(V, name="Temperature")
v = ufl.TestFunction(V)

# dirichlet 30
facet_indices = np.arange(num_facets, dtype=np.int64)
facet_tags = meshtags(domain_mesh, fdim, facet_indices, markers)
bc = fem.dirichletbc(PETSc.ScalarType(30), fem.locate_dofs_topological(V, fdim, facet_tags.find(2)), V)
bcs = []

ds = ufl.Measure("ds", domain=domain_mesh, subdomain_data=facet_tags)

kappa = fem.Constant(domain_mesh, PETSc.ScalarType(3))

x = ufl.SpatialCoordinate(domain_mesh)

xc, yc = (x0 + dx_/2), (y0 + dy_/2)
wx, wy  = 0.50, 0.50

xl, xr = xc - wx/2, xc + wx/2
yl, yr = yc - wy/2, yc + wy/2

a = 60.0
source_grad = kappa*100

step_lx = 0.5*(ufl.tanh(a*(x[0] - xl)) + 1.0)
step_rx = 0.5*(ufl.tanh(a*(xr - x[0])) + 1.0)
step_ly = 0.5*(ufl.tanh(a*(x[1] - yl)) + 1.0)
step_ry = 0.5*(ufl.tanh(a*(yr - x[1])) + 1.0)

indicator = step_lx * step_rx * step_ly * step_ry
q_flux    = -source_grad * indicator

h_conv = 50
T_amb = 30

n = ufl.FacetNormal(domain_mesh)
f = 1
x = ufl.SpatialCoordinate(domain_mesh)
F = kappa*ufl.dot(ufl.grad(T), ufl.grad(v)) * ufl.dx
F += q_flux * v * ds(1)
F += h_conv * (T - T_amb) * v * ds(2)
F += f * v * ufl.dx

problem = NonlinearProblem(F, T, bcs=bcs)

solver = NewtonSolver(MPI.COMM_WORLD, problem)
solver.convergence_criterion = "incremental"

niter, converged = solver.solve(T)
T.x.scatter_forward()

path_to_sol = os.path.join(BASE_DIR, "heat_box/temperature_solution.vtk")

T_true = fem.Function(V, name="Temperature_true")

writer = io.VTKFile(MPI.COMM_WORLD, path_to_sol, "w")
writer.write_function([T, T_true], 0.0)
writer.write_mesh(domain_mesh)

print("FEniCSx solve complete.")
grid = pv.UnstructuredGrid(*plot.vtk_mesh(V))
grid.point_data["uh"] = T.x.array
grid.point_data["uh_1"] = np.zeros_like(T.x.array)
plotter = pv.Plotter()
viridis = plt.colormaps.get_cmap("coolwarm").resampled(25)

renderer = plotter.add_mesh(grid, show_scalar_bar=True, cmap=viridis)
plotter.camera_position = 'xy'

plotter.show()
