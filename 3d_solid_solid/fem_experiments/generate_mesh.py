#!/usr/bin/env python3
"""
Generate a 3‑D box with a 2‑D heat‑source patch on its bottom face,
mesh it with Gmsh, convert to Dolfin‑X and write the result to disk.

Author: <your‑name>
"""

from mpi4py import MPI
from dolfinx.io import gmshio, XDMFFile
import gmsh
from pathlib import Path
from typing import List, Tuple
import numpy as np

comm  = MPI.COMM_WORLD
rank  = comm.rank
root  = 0

# -----------------------------------------------------------------------------#
# Utility helpers
# -----------------------------------------------------------------------------#
tol: float = 1e-6

def surfaces_at_z(model, z: float,
                  x_min=None, x_max=None,
                  y_min=None, y_max=None) -> List[int]:
    """Return tags of planar surfaces lying in z = constant (within tol)."""
    ids: List[int] = []
    for (dim, tag) in model.getEntities(2):
        xmin, ymin, zmin, xmax, ymax, zmax = model.getBoundingBox(dim, tag)
        if abs(zmin - z) < tol and abs(zmax - z) < tol:
            if x_min is not None and xmin > x_min + tol:
                continue
            if x_max is not None and xmax < x_max - tol:
                continue
            if y_min is not None and ymin > y_min + tol:
                continue
            if y_max is not None and ymax < y_max - tol:
                continue
            ids.append(tag)
    return ids

# -----------------------------------------------------------------------------#
# Geometry definition (rank 0 only)
# -----------------------------------------------------------------------------#

lc  = 0.03

if rank == root:
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", lc)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", lc)

    model, occ = gmsh.model, gmsh.model.occ
    model.add("box_with_bottom_heat_points")

    # Box parameters
    dx, dy, dz = 0.65, 0.875, 0.05
    x0, y0, z0 = -0.5*dx, -0.5*dy, -0.5*dz

    # Create base box
    base_tag = occ.addBox(x0, y0, z0, dx, dy, dz)
    occ.synchronize()

    # Three fins on top face
    nfins, fin_w, fin_h = 5, 0.05, 0.8625
    # compute the spacing between fins so that they exactly span the top face
    if nfins > 1:
        inter_gap = (dy - nfins*fin_w)/(nfins - 1)
    else:
        inter_gap = 0.0
    z_top  = z0 + dz
    fin_tags: List[int] = []
    for i in range(nfins):
        # first fin at y=y0, last fin at y=y0+dy−fin_w
        y_start = y0 + i*(fin_w + inter_gap)
        ft = occ.addBox(x0, y_start, z_top, dx, fin_w, fin_h)
        fin_tags.append(ft)
    occ.synchronize()

    # Fuse base and fins
    fused, _ = occ.fuse([(3, base_tag)], [(3, ft) for ft in fin_tags], removeObject=True, removeTool=True)
    occ.synchronize()
    fused_vol = [tag for (dim, tag) in fused if dim == 3][0]

    # Identify bottom face surfaces
    bottom_faces = surfaces_at_z(model, z0)

    # relative positions in x and y within bottom_faces extents
    heat_point_tags: List[int] = []
    # compute bounding box of first bottom face
    xmin_bf, ymin_bf, _, xmax_bf, ymax_bf, _ = model.getBoundingBox(2, bottom_faces[0])
    rels: List[Tuple[float, float]] = [(0.25, 0.25), (0.25, 0.75), (0.75, 0.25), (0.75, 0.75)]
    for rx, ry in rels:
        x_pt = xmin_bf + rx * (xmax_bf - xmin_bf)
        y_pt = ymin_bf + ry * (ymax_bf - ymin_bf)
        ptag = occ.addPoint(x_pt, y_pt, z0)
        heat_point_tags.append(ptag)
    occ.synchronize()

    # Fragment the bottom face by these points
    object_surfs = [(2, s) for s in bottom_faces]
    tool_pts     = [(0, p) for p in heat_point_tags]
    frags, _ = occ.fragment(object_surfs, tool_pts)
    occ.synchronize()

    # Recollect bottom sub-surfaces
    new_bottom = surfaces_at_z(model, z0)

    # Partition into heat-point patches vs rest
    heat_surfs = []
    rest_surfs = []
    for s in new_bottom:
        xmin, ymin, _, xmax, ymax, _ = model.getBoundingBox(2, s)
        if any(xmin - tol <= xmin_bf + rx*(xmax_bf-xmin_bf) <= xmax + tol and
               ymin - tol <= ymin_bf + ry*(ymax_bf-ymin_bf) <= ymax + tol
               for rx, ry in rels):
            heat_surfs.append(s)

    # Physical groups
    model.addPhysicalGroup(3, [fused_vol], tag=1)
    model.setPhysicalName(3, 1, "BoxWithFins")

    model.addPhysicalGroup(2, heat_surfs, tag=6)
    model.setPhysicalName(2, 6, "BottomPlate")

    # Side walls and fin tops
    all_surfs = model.getBoundary([(3, fused_vol)], oriented=False, recursive=False)
    all_surfs = [s[1] for s in all_surfs]
    top_fin_faces = surfaces_at_z(model, z_top + fin_h)
    side_surfs = [s for s in all_surfs if s not in heat_surfs + top_fin_faces + new_bottom]
    model.addPhysicalGroup(2, side_surfs, tag=4)
    model.setPhysicalName(2, 4, "SideWalls")

    model.addPhysicalGroup(2, top_fin_faces, tag=5)
    model.setPhysicalName(2, 5, "FinTop")

    # Mesh and write files
    model.mesh.generate(3)

    outdir = (Path(__file__).resolve().parent
              if "__file__" in globals() else Path.cwd())
    msh_path = outdir / "box_heat_source.msh"
    gmsh.write(str(msh_path))

else:
    model = None   # non‑root ranks receive the mesh from root

# -----------------------------------------------------------------------------#
# Convert to Dolfin‑X and write XDMF
# -----------------------------------------------------------------------------#
mesh, ct, ft = gmshio.model_to_mesh(model, comm, root, gdim=3)

if rank == root:
    gmsh.finalize()

xdmf_path = (Path(__file__).resolve().parent
             if "__file__" in globals() else Path.cwd()) / "box_heat_source.xdmf"

with XDMFFile(comm, str(xdmf_path), "w") as xdmf:
    mesh.topology.create_connectivity(2, 3)   # needed for facet data
    xdmf.write_mesh(mesh)
    xdmf.write_meshtags(ct, mesh.geometry)
    xdmf.write_meshtags(ft, mesh.geometry)

if rank == root:
    print(f"Mesh written to: {msh_path}")
    print(f"XDMF written to: {xdmf_path}")
