import os
import sys
import warnings

import torch
import numpy as np
from sympy import Symbol, Eq, And, Or, tanh, Not
from typing import Dict
import physicsnemo.sym
from physicsnemo.sym.hydra import to_absolute_path, instantiate_arch
from physicsnemo.sym.solver import Solver
from physicsnemo.sym.domain import Domain
from physicsnemo.sym.geometry.primitives_2d import Rectangle, Line, Channel2D, Polygon
from physicsnemo.sym.geometry.primitives_3d import Box, Channel, Plane
from physicsnemo.sym.utils.sympy.functions import parabola
from physicsnemo.sym.utils.io import csv_to_dict
from physicsnemo.sym.eq.pdes.navier_stokes import NavierStokes, GradNormal
from physicsnemo.sym.eq.pdes.diffusion import Diffusion, DiffusionInterface
from physicsnemo.sym.eq.pdes.basic import NormalDotVec
from physicsnemo.sym.eq.pdes.turbulence_zero_eq import ZeroEquation
from physicsnemo.sym.eq.pdes.advection_diffusion import AdvectionDiffusion
from physicsnemo.sym.domain.constraint import (
    PointwiseBoundaryConstraint,
    PointwiseInteriorConstraint,
    IntegralBoundaryConstraint,
)
from physicsnemo.sym.domain.monitor import PointwiseMonitor
from physicsnemo.sym.domain.validator import PointwiseValidator, PointVTKValidator, GridValidator
from physicsnemo.sym.key import Key
from physicsnemo.sym.node import Node
from physicsnemo.sym.geometry import Parameterization, Parameter
from physicsnemo.sym.models.fully_connected import FullyConnectedArch
from physicsnemo.models.layers.activations import get_activation
from physicsnemo.sym.utils.io.vtk import VTKFromFile, VTKUniformGrid
from physicsnemo.sym.domain.inferencer import PointVTKInferencer
from physicsnemo.sym.utils.io import InferencerPlotter, GridValidatorPlotter, plotter
from physicsnemo.sym.models.modified_fourier_net import ModifiedFourierNetArch
from physicsnemo.sym.models.fourier_net import FourierNetArch
from physicsnemo.sym.utils.io.vtk import var_to_polyvtk
from conv_bc import ConvectiveBC
from vtk_plotter import VTKPlotter
import matplotlib.pyplot as plt

dir_path = os.path.dirname(os.path.abspath(__file__))

@physicsnemo.sym.main(config_path="conf", config_name="config")
def run(cfg) -> None:
    cfg.network_dir = dir_path + "/outputs/fixed/" + cfg.custom.network + f"_{cfg.custom.layer_size}_{cfg.custom.activation}"
    cfg.initialization_network_dir = dir_path + "/outputs/fixed/" + cfg.custom.network + f"_{cfg.custom.layer_size}_{cfg.custom.activation}"
    dx, dy, dz = 0.65, 0.875, 0.05
    x0, y0, z0 = -dx / 2, -dy / 2, -dz / 2

    plotter = VTKPlotter(
        path_to_pinns=f"{cfg.network_dir}/inferencers/vtk_inf.vtu",
        path_to_vtk=dir_path + "/temp_sol.vtu",
        slice_origins=[(0, 0, -0.02449326630430), (0, 0, 0)],
        slice_normals=[(0, 0, 1), (0, 1, 0)],
        array_name="Temperature_true"
    )

    kappa = 3
    ambient_temp = 30
    h_conv = 10
    L = dx
    delta_t = 100 * L / kappa
    Bi = h_conv * L / kappa

    # bottom‐patch fraction
    hx_frac, hy_frac = 0.50, 0.50
    hx, hy = dx * hx_frac, dy * hy_frac
    hs_x0 = x0 + 0.5 * (dx - hx)
    hs_y0 = y0 + 0.5 * (dy - hy)

    # fins on top
    nfins, fin_w, fin_h = 17, 0.0075, 0.8625
    gap = (dy - nfins * fin_w) / (nfins - 1) if nfins > 1 else 0.0

    # base plate
    plate = Box((x0/L, y0/L, z0/L), ((x0 + dx)/L, (y0 + dy)/L, (z0 + dz)/L))

    # fins, first fin at y=y0, last at y=y0+dy−fin_w
    single_fin = Box((x0/L, y0/L, (z0 + dz)/L), ((x0 + dx)/L, (y0 + fin_w)/L, (z0 + dz + fin_h)/L))
    fin_center = ((x0 + x0 + dx) / (2*L), (y0 + y0 + fin_w) / (2*L), z0 + dz + fin_h / (2*L))
    fins = single_fin.repeat(
        (gap + fin_w)/L,
        repeat_lower=(0,0,0),
        repeat_higher=(0, nfins - 1, 0),
        center=fin_center
    )

    heat_sink = plate + fins

    # s = heat_sink.sample_boundary(nr_points=10000)
    # var_to_polyvtk(s, "boundary")
    # print("Surface Area:{:.3f}".format(np.sum(s["area"])))
    # s = heat_sink.sample_interior(nr_points=10000, compute_sdf_derivatives=True)
    # var_to_polyvtk(s, "interior")

    domain = Domain()

    heat_eq = Diffusion(T="theta_star", D=1.0, dim=3, Q=0.0, time=False)
    grad_theta = GradNormal("theta_star", dim=3, time=False)
    conv_theta = ConvectiveBC(
        T="theta_star", kappa=1.0, h=Bi, T_ext=ambient_temp, dim=3, time=False, non_dim=True
    )
    x_ref  = -dx/2
    y_ref  = -dy/2
    z_ref  = -dz/2

    input_keys = [Key("x"), Key("y"), Key("z")]
    if cfg.custom.network == "fully_connected":
        heat_net = FullyConnectedArch(
            input_keys=input_keys,
            output_keys=[Key("theta_star")],
            layer_size=cfg.custom.layer_size,
            activation_fn=get_activation(cfg.custom.activation),
        )
    elif cfg.custom.network == "fourier_net":
        heat_net = FourierNetArch(
            input_keys=input_keys,
            output_keys=[Key("theta_star")],
            layer_size=cfg.custom.layer_size,
            activation_fn=get_activation(cfg.custom.activation),
        )
    elif cfg.custom.network == "modified_fourier_net":
        heat_net = ModifiedFourierNetArch(
            input_keys=input_keys,
            output_keys=[Key("theta_star")],
            layer_size=cfg.custom.layer_size,
            activation_fn=get_activation(cfg.custom.activation),
        )
    else:
        sys.exit(
            f"Unknown network type {cfg.custom.network}. Please choose 'fully_connected' or 'fourier_net'."
        )
    phys_node = [
        Node.from_sympy(delta_t * Symbol("theta_star") + ambient_temp, "theta_phys")
    ]
    x_star = Symbol("x")
    y_star = Symbol("y")
    z_star = Symbol("z")
    x_phys_node = Node.from_sympy(x_star * L + x_ref, "x_phys")
    y_phys_node = Node.from_sympy(y_star * L + y_ref, "y_phys")
    z_phys_node = Node.from_sympy(z_star * L + z_ref, "z_phys")
    coordinates = [x_phys_node, y_phys_node, z_phys_node]
    nodes = (
        coordinates +
        heat_eq.make_nodes() +
        grad_theta.make_nodes() +
        conv_theta.make_nodes() +
        [heat_net.make_node(name="heat_net")] +
        phys_node
    )

    x, y, z = Symbol("x"), Symbol("y"), Symbol("z")

    interior = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=heat_sink,
        outvar={"diffusion_theta_star": 0},
        batch_size=cfg.batch_size.interior
    )
    domain.add_constraint(interior, "interior")

    source_grad = 1

    # xc, yc = (x0 + dx/2), (y0 + dy/2)
    xc, yc = (x0 + dx/2)/L, (y0 + dy/2)/L
    wx, wy  = 0.25, 0.25

    xl, xr = xc - wx/2, xc + wx/2
    yl, yr = yc - wy/2, yc + wy/2

    a = 60.0

    step_lx = 0.5*(tanh(a*(x - xl)) + 1.0)
    step_rx = 0.5*(tanh(a*(xr - x)) + 1.0)
    step_ly = 0.5*(tanh(a*(y - yl)) + 1.0)
    step_ry = 0.5*(tanh(a*(yr - y)) + 1.0)

    indicator = step_lx * step_rx * step_ly * step_ry
    gradient_normal = source_grad * indicator

    z0_dimless = z0 / L
    heat_source = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=heat_sink,
        outvar={"normal_gradient_theta_star": gradient_normal},
        batch_size=cfg.batch_size.heat_source,
        criteria=(Eq(z, z0_dimless)),
    )
    domain.add_constraint(heat_source, "heat_source")

    convective = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=heat_sink,
        outvar={"convective_theta_star": 0},
        batch_size=cfg.batch_size.boundary,
        # lambda_weighting={"convective_theta": walls_sdf},
        criteria=Not(Eq(z, z0_dimless)),
    )
    domain.add_constraint(convective, "convective")

    vtk_obj = VTKFromFile(
        file_path=to_absolute_path(dir_path + "/temp_sol.vtu"),
        export_map={"Temperature": ["theta_star"], "Temperature_true": ["theta_phys"]},	
    )

    grid_inferencer = PointVTKInferencer(
        vtk_obj=vtk_obj,
        nodes=nodes,
        input_vtk_map={"x": "x", "y": "y", "z": "z"},
        output_names=["theta_star", "theta_phys"],
        batch_size=1024,
        requires_grad=False,
    )
    domain.add_inferencer(grid_inferencer, "vtk_inf")

    grid_validator = PointVTKValidator(
        vtk_obj=vtk_obj,
        nodes=nodes,
        input_vtk_map={"x": "x", "y": "y", "z": "z"},
        true_vtk_map={"theta_phys": ["Temperature_true"]},
        requires_grad=False,
        batch_size=1024,
        plotter=plotter
    )
    domain.add_validator(grid_validator, "vtk_val")

    # make solver
    slv = Solver(cfg, domain)

    # start solver
    slv.solve()


if __name__ == "__main__":
    run()
