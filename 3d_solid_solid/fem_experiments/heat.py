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

domain_mesh, cell_tags, facet_tags = gmshio.read_from_msh(
    BASE_DIR + "/box_heat_source.msh", comm, gdim=3
)

fdim = domain_mesh.topology.dim - 1

V = fem.functionspace(domain_mesh, ("Lagrange", 1))

dx = ufl.Measure("dx", domain=domain_mesh, subdomain_data=cell_tags)
ds = ufl.Measure("ds", domain=domain_mesh, subdomain_data=facet_tags)

T = fem.Function(V, name="Temperature")
v = ufl.TestFunction(V)

dx_, dy_, dz_ = 0.65, 0.875, 0.05
x0, y0, z0 = -0.5*dx_, -0.5*dy_, -0.5*dz_
hx_frac, hy_frac = 0.50, 0.50
hx, hy = dx_*hx_frac, dy_*hy_frac
hs_x0 = x0 + 0.5*(dx_ - hx)
hs_y0 = y0 + 0.5*(dy_ - hy)
# def heat_source(x):
#     return (
#          np.isclose(x[2], z0) &
#          (x[0] >= hs_x0) & (x[0] <= hs_x0 + hx) &
#         (x[1] >= hs_y0) & (x[1] <= hs_y0 + hy)
#     )

# num_facets = domain_mesh.topology.index_map(fdim).size_local
# markers = np.zeros(num_facets, dtype=np.int64)
# heat_source_facets = locate_entities_boundary(domain_mesh, fdim, heat_source)

# side_wall_facets = facet_tags.find(4)
# fin_top_facets = facet_tags.find(5)

# markers[heat_source_facets] = 11
# markers[side_wall_facets] = 4
# markers[fin_top_facets] = 5

# facet_indices = np.arange(num_facets, dtype=np.int64)
# facet_tags = meshtags(domain_mesh, fdim, facet_indices, markers)

# # now define a new ds over that tag‑collection
# facet_tags = meshtags(domain_mesh, fdim,
#                       np.arange(num_facets, dtype=np.int32),
#                       markers)
# ds = ufl.Measure("ds", domain=domain_mesh, subdomain_data=facet_tags)
bc = fem.dirichletbc(PETSc.ScalarType(100), fem.locate_dofs_topological(V, fdim, facet_tags.find(11)), V)

bcs = []

kappa = fem.Constant(domain_mesh, PETSc.ScalarType(3))

x = ufl.SpatialCoordinate(domain_mesh)

xc, yc = (x0 + dx_/2), (y0 + dy_/2)
wx, wy  = 0.25, 0.25

xl, xr = xc - wx/2, xc + wx/2
yl, yr = yc - wy/2, yc + wy/2

a = 60.0
source_grad = 100 * kappa

step_lx = 0.5*(ufl.tanh(a*(x[0] - xl)) + 1.0)
step_rx = 0.5*(ufl.tanh(a*(xr - x[0])) + 1.0)
step_ly = 0.5*(ufl.tanh(a*(x[1] - yl)) + 1.0)
step_ry = 0.5*(ufl.tanh(a*(yr - x[1])) + 1.0)

indicator = step_lx * step_rx * step_ly * step_ry
q_flux    = -source_grad * indicator

n = ufl.FacetNormal(domain_mesh)
h_conv = 1
T_amb = 30
f = 1
x = ufl.SpatialCoordinate(domain_mesh)
F = kappa*ufl.dot(ufl.grad(T), ufl.grad(v)) * dx
F += h_conv*(T - T_amb) * v * ds(4)
F += h_conv*(T - T_amb) * v * ds(5)
F += q_flux * v * ds(6)
F += f * v * dx

a = ufl.lhs(F)
L = ufl.rhs(F)

problem = NonlinearProblem(F, T, bcs=bcs)

solver = NewtonSolver(MPI.COMM_WORLD, problem)
solver.convergence_criterion = "incremental"

niter, converged = solver.solve(T)
T.x.scatter_forward()

path_to_sol = os.path.join(BASE_DIR, "temperature_solution.vtk")

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
