"""
Test for heat diffusion in a 3D box with a heat source in the center
"""
import os
import sys
import warnings

import torch
import numpy as np
from sympy import Symbol, Eq, And, Or, tanh, Not

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
from physicsnemo.sym.domain.validator import PointwiseValidator, PointVTKValidator
from physicsnemo.sym.key import Key
from physicsnemo.sym.node import Node
from physicsnemo.sym.geometry import Parameterization, Parameter
from physicsnemo.sym.models.fully_connected import FullyConnectedArch
from physicsnemo.models.layers.activations import get_activation
from physicsnemo.sym.utils.io.vtk import VTKFromFile, VTKUniformGrid
from physicsnemo.sym.domain.inferencer import PointVTKInferencer
from physicsnemo.sym.utils.io import InferencerPlotter, GridValidatorPlotter, GridValidatorPlotter
from physicsnemo.sym.models.modified_fourier_net import ModifiedFourierNetArch
from physicsnemo.sym.models.fourier_net import FourierNetArch
from physicsnemo.sym.utils.io.vtk import var_to_polyvtk
from conv_bc import ConvectiveBC
from vtk_plotter import VTKPlotter

dir_path = os.path.dirname(os.path.abspath(__file__))


@physicsnemo.sym.main(config_path="conf", config_name="config")
def run(cfg) -> None:
    cfg.network_dir = dir_path + "/heat_box_outputs/" + cfg.custom.network + f"_{cfg.custom.layer_size}_{cfg.custom.activation}"
    cfg.initialization_network_dir = dir_path + "/heat_box_outputs/" + cfg.custom.network + f"_{cfg.custom.layer_size}_{cfg.custom.activation}"

    x0, y0, z0 = -1.0, -1.0, -1.0
    dx, dy, dz = 2.0, 2.0, 2.0
    # unit box
    box = Box(
        point_1=(x0, y0, z0),
        point_2=(x0 + dx, y0 + dy, z0 + dz)
    )
    kappa = 3
    h_conv = 50
    ambient_temp = 30
    domain = Domain()

    heat_eq = Diffusion(T="theta", D=kappa, dim=3, Q=1.0, time=False)
    grad_theta = GradNormal("theta", dim=3, time=False)
    conv_theta = ConvectiveBC(
        T="theta", kappa=kappa, h=h_conv, T_ext=ambient_temp, dim=3, time=False
    )

    input_keys = [Key("x"), Key("y"), Key("z")]
    if cfg.custom.network == "fully_connected":
        heat_net = FullyConnectedArch(
            input_keys=input_keys,
            output_keys=[Key("theta")],
            layer_size=cfg.custom.layer_size,
            activation_fn=get_activation(cfg.custom.activation),
        )
    elif cfg.custom.network == "fourier_net":
        heat_net = FourierNetArch(
            input_keys=input_keys,
            output_keys=[Key("theta")],
            layer_size=cfg.custom.layer_size,
            activation_fn=get_activation(cfg.custom.activation),
        )
    else:
        sys.exit(
            f"Unknown network type {cfg.custom.network}. Please choose 'fully_connected' or 'fourier_net'."
        )
    # phys_node = [
    #     Node.from_sympy(delta_t * Symbol("theta") + ambient_temp, "theta_phys")
    # ]
    nodes = (
        heat_eq.make_nodes() +
        grad_theta.make_nodes() +
        conv_theta.make_nodes() +
        [heat_net.make_node(name="heat_net")]
        # phys_node
    )

    x, y, z = Symbol("x"), Symbol("y"), Symbol("z")

    interior = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=box,
        outvar={"diffusion_theta": 0},
        batch_size=cfg.batch_size.interior,
        lambda_weighting={"diffusion_theta": Symbol("sdf")},
    )
    domain.add_constraint(interior, "interior")

    source_grad = 100

    xc, yc = (x0 + dx/2), (y0 + dy/2)
    wx, wy  = 0.50, 0.50

    xl, xr = xc - wx/2, xc + wx/2
    yl, yr = yc - wy/2, yc + wy/2

    a = 60.0

    step_lx = 0.5*(tanh(a*(x - xl)) + 1.0)
    step_rx = 0.5*(tanh(a*(xr - x)) + 1.0)
    step_ly = 0.5*(tanh(a*(y - yl)) + 1.0)
    step_ry = 0.5*(tanh(a*(yr - y)) + 1.0)

    indicator = step_lx * step_rx * step_ly * step_ry
    gradient_normal = source_grad * indicator

    heat_source = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=box,
        outvar={"normal_gradient_theta": gradient_normal},
        batch_size=cfg.batch_size.heat_source,
        criteria=(Eq(z, z0)),
        batch_per_epoch=1000,
    )
    domain.add_constraint(heat_source, "heat_source")

    convective = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=box,
        outvar={"convective_theta": 0},
        batch_size=cfg.batch_size.boundary,
        criteria=Not(Eq(z, z0)),
    )
    domain.add_constraint(convective, "convective")

    vtk_obj = VTKFromFile(
        file_path=to_absolute_path(dir_path + "/heat_box.vtu"),
        export_map={"Temperature": ["theta"]},
    )

    grid_inferencer = PointVTKInferencer(
        vtk_obj=vtk_obj,
        nodes=nodes,
        input_vtk_map={"x": "x", "y": "y", "z": "z"},
        output_names=["theta"],
        batch_size=1024,
        requires_grad=False,
    )
    domain.add_inferencer(grid_inferencer, "vtk_inf")

    plotter = VTKPlotter(
        path_to_pinns=f"{cfg.network_dir}/inferencers/vtk_inf.vtu",
        path_to_vtk=dir_path + "/heat_box.vtu",
        slice_origins=[(0, 0, -0.99), (0, 0, 0)],
        slice_normals=[(0, 0, 1), (0, 1, 0)],
        array_name="Temperature"
    )

    grid_validator = PointVTKValidator(
        vtk_obj=vtk_obj,
        nodes=nodes,
        input_vtk_map={"x": "x", "y": "y", "z": "z"},
        true_vtk_map={"theta": ["Temperature"]},
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
