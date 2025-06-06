import numpy as np
import os
import ufl
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import fem, mesh, io, plot
from dolfinx.fem.petsc import LinearProblem, NonlinearProblem
from dolfinx.nls.petsc import NewtonSolver
from dolfinx.io import XDMFFile, VTKFile
import pyvista as pv
from dolfinx.mesh import meshtags, locate_entities_boundary

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
width = 2  # m
height = 1  # m
T_heat = 1173  # K
T_amb = 323  # K
k_thermal = 3.0  # [W/(m·K)] thermal conductivity
h_conv = 50.0  # [W/(m^2·K)] convection coefficient
epsilon = 0.7  # emissivity
sigma_SB = 5.670374419e-8  # [W/(m^2·K^4)] Stefan-Boltzmann constant
nx, ny = 40, 40
domain = mesh.create_rectangle(
    MPI.COMM_WORLD,
    [np.array([0, 0]), np.array([width, height])],
    [nx, ny],
    mesh.CellType.triangle,
)

tdim = domain.topology.dim
fdim = tdim - 1  # dimension of boundary facets


def on_left(x):
    return np.isclose(x[0], 0.0)


def on_right(x):
    return np.isclose(x[0], width)


def on_top(x):
    return np.isclose(x[1], height)


def on_bottom(x):
    return np.isclose(x[1], 0.0)


left_facets = locate_entities_boundary(domain, fdim, on_left)
right_facets = locate_entities_boundary(domain, fdim, on_right)
top_facets = locate_entities_boundary(domain, fdim, on_top)
bottom_facets = locate_entities_boundary(domain, fdim, on_bottom)

num_facets = domain.topology.index_map(fdim).size_local
markers = np.zeros(num_facets, dtype=np.int64)

markers[left_facets] = 1  # Dirichlet
markers[top_facets] = 2  # Radiative
markers[right_facets] = 1  # Dirichlet
markers[bottom_facets] = 4  # Isolated

facet_indices = np.arange(num_facets, dtype=np.int64)
facet_tags = meshtags(domain, fdim, facet_indices, markers)

V = fem.functionspace(domain, ("Lagrange", 2))

T = fem.Function(V, name="Temperature")
v = ufl.TestFunction(V)

ds = ufl.Measure("ds", domain=domain, subdomain_data=facet_tags)

T_hot_const = fem.Constant(domain, PETSc.ScalarType(T_heat))

dirichlet_facets = np.where(markers == 1)[0]
dirichlet_dofs = fem.locate_dofs_topological(V, fdim, facet_indices[dirichlet_facets])
bc_dirichlet = fem.dirichletbc(T_hot_const, dirichlet_dofs, V)
bcs = [bc_dirichlet]

diffusion = k_thermal * ufl.dot(ufl.grad(T), ufl.grad(v)) * ufl.dx

flux_expr = h_conv * (T - T_amb) + epsilon * sigma_SB * (T**4 - T_amb**4)
flux_bc = flux_expr * v * ds(2)

F = diffusion + flux_bc

problem = NonlinearProblem(F, T, bcs=bcs)

solver = NewtonSolver(MPI.COMM_WORLD, problem)
solver.convergence_criterion = "incremental"
solver.rtol = 1e-10
solver.atol = 1e-10
solver.report = True

niter, converged = solver.solve(T)
T.x.scatter_forward()

# q-gain, q-loss
n = ufl.FacetNormal(domain)
q_conv_form = h_conv * (T - T_amb)
q_conv = fem.assemble_scalar(fem.form(q_conv_form * ds(2)))

q_rad_form = epsilon * sigma_SB * (T**4 - T_amb**4)
q_rad = fem.assemble_scalar(fem.form(q_rad_form * ds(2)))

# Total heat loss at top
q_loss_top = q_conv + q_rad
print(f"q_loss_top = {q_loss_top}")
q_cond_form = k_thermal * ufl.dot(ufl.grad(T), n)
q_cond_sides = fem.assemble_scalar(fem.form(q_cond_form * ds(1)))
print(f"q_cond_sides = {q_cond_sides}")

rank = MPI.COMM_WORLD.rank
if rank == 0:
    print(f"Newton solver finished in {niter} iterations. Converged = {converged}")

V_linear = fem.functionspace(domain, ("Lagrange", 1))
T_linear = fem.Function(V_linear, name="Temperature")
T_linear.interpolate(T)

path_to_sol = os.path.join(BASE_DIR, "temperature_solution.vtu")
writer = io.VTKFile(MPI.COMM_WORLD, path_to_sol, "w")
writer.write_function([T], 0.0)
writer.write_mesh(domain)
