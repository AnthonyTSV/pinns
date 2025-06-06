import numpy as np
import os
import ufl
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import fem, mesh, io, plot
from dolfinx.fem.petsc import LinearProblem
from dolfinx.io import XDMFFile, VTKFile
import pyvista as pv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
Lx, Ly = 1, 1
nx, ny = 25, 25
domain = mesh.create_rectangle(MPI.COMM_WORLD, [np.array([0, 0]), np.array([Lx, Ly])], [nx, ny], mesh.CellType.triangle)

V = fem.functionspace(domain, ("Lagrange", 2))
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

def left_boundary(x):
    return np.isclose(x[0], 0.0)

def right_boundary(x):
    return np.isclose(x[0], Lx)

def top_boundary(x):
    return np.isclose(x[1], Ly)

def bottom_boundary(x):
    return np.isclose(x[1], 0.0)

u_left = fem.Constant(domain, PETSc.ScalarType(25))
u_right = fem.Constant(domain, PETSc.ScalarType(5))

left_dofs = fem.locate_dofs_geometrical(V, left_boundary)
right_dofs = fem.locate_dofs_geometrical(V, right_boundary)
top_dofs = fem.locate_dofs_geometrical(V, top_boundary)
bottom_dofs = fem.locate_dofs_geometrical(V, bottom_boundary)

bc_left = fem.dirichletbc(u_left, left_dofs, V)
bc_right = fem.dirichletbc(u_right, right_dofs, V)
bc_top = fem.dirichletbc(u_right, top_dofs, V)
bc_bottom = fem.dirichletbc(u_right, bottom_dofs, V)
bcs = [bc_left, bc_right, bc_top, bc_bottom]

a = ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
L = v * ufl.dx
problem = LinearProblem(a, L, bcs=bcs, petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
u_sol = problem.solve()

path_to_sol = os.path.join(BASE_DIR, "temperature_solution.vtk")

writer = io.VTKFile(MPI.COMM_WORLD, path_to_sol, "w")
writer.write_function(u_sol, 0.0)
writer.write_mesh(domain)


print("FEniCSx solve complete.")
grid = pv.UnstructuredGrid(*plot.vtk_mesh(V))
grid.point_data["uh"] = u_sol.x.array
plotter = pv.Plotter()
viridis = plt.colormaps.get_cmap("viridis").resampled(5)

renderer = plotter.add_mesh(grid, show_scalar_bar=True, cmap=viridis, show_edges=True)

plotter.show()