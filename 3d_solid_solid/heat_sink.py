import os
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
from physicsnemo.sym.utils.io import InferencerPlotter, GridValidatorPlotter
from physicsnemo.sym.models.modified_fourier_net import ModifiedFourierNetArch
from physicsnemo.sym.models.fourier_net import FourierNetArch
from physicsnemo.sym.utils.io.vtk import var_to_polyvtk
from conv_bc import ConvectiveBC

dir_path = os.path.dirname(os.path.abspath(__file__))


@physicsnemo.sym.main(config_path="conf", config_name="config")
def run(cfg) -> None:
    cfg.network_dir = dir_path + "/outputs"
    cfg.initialization_network_dir = dir_path + "/outputs"
    x0, y0, z0 = -0.75, -0.50, -0.4375
    dx, dy, dz = 0.2, 0.2, 0.01

    kappa = 1
    source_temp = 100
    ambient_temp = 30
    h_conv = 0.1

    delta_t = source_temp - ambient_temp

    # bottom‐patch fraction
    hx_frac, hy_frac = 0.50, 0.50
    hx, hy = dx * hx_frac, dy * hy_frac
    hs_x0 = x0 + 0.5 * (dx - hx)
    hs_y0 = y0 + 0.5 * (dy - hy)

    # fins on top
    nfins, fin_w, fin_h = 5, 0.005, 0.2
    gap = (dy - nfins * fin_w) / (nfins - 1) if nfins > 1 else 0.0

    # base plate
    plate = Box((x0, y0, z0), (x0 + dx, y0 + dy, z0 + dz))

    # fins, first fin at y=y0, last at y=y0+dy−fin_w
    single_fin = Box((x0, y0, z0 + dz), (x0 + dx, y0 + fin_w, z0 + dz + fin_h))
    fin_center = ((x0 + x0 + dx) / 2, (y0 + y0 + fin_w) / 2, z0 + dz + fin_h / 2)
    fins = single_fin.repeat(
        gap + fin_w,
        repeat_lower=(0,0,0),
        repeat_higher=(0, nfins - 1, 0),
        center=fin_center
    )

    heat_sink = plate + fins

    domain = Domain()

    heat_eq = Diffusion(T="theta", D=1.0, dim=3, Q=1.0, time=False)
    grad_theta = GradNormal("theta", dim=3, time=False)
    conv_theta = ConvectiveBC(
        T="theta", kappa=kappa, h=h_conv, T_ext=ambient_temp, dim=3, time=False
    )

    input_keys = [Key("x"), Key("y"), Key("z")]
    heat_net = ModifiedFourierNetArch(
        input_keys=input_keys,
        layer_size=128,
        output_keys=[Key("theta_star")],
        activation_fn=get_activation("tanh"),
    )
    phys_node = [
        Node.from_sympy(delta_t * Symbol("theta_star") + ambient_temp, "theta")
    ]
    nodes = (
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
        outvar={"diffusion_theta": 0},
        batch_size=cfg.batch_size.interior,
        lambda_weighting={
            "diffusion_theta": Symbol("sdf"),
        },
    )
    domain.add_constraint(interior, "interior")

    heat_source = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=heat_sink,
        outvar={"theta": source_temp},
        batch_size=cfg.batch_size.heat_source,
        lambda_weighting={"theta": 10},
        criteria=
            Eq(z, z0)
            & (x >= hs_x0)
            & (x <= hs_x0 + hx)
            & (y >= hs_y0)
            & (y <= hs_y0 + hy)
    )
    domain.add_constraint(heat_source, "heat_source")

    bottom_rest = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=heat_sink,
        outvar={"normal_gradient_theta": 0},
        batch_size=cfg.batch_size.boundary,
        criteria=(
            Eq(z, z0)
            & Or(x < hs_x0, x > hs_x0 + hx,
                y < hs_y0, y > hs_y0 + hy)
        )
    )
    domain.add_constraint(bottom_rest, "bottom_rest")

    def walls_sdf(x, y, z):
        sdf = heat_sink.sdf({"x": x, "y": y, "z": z}, {})
        return sdf["sdf"]

    convective = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=heat_sink,
        outvar={"convective_theta": 0},
        batch_size=cfg.batch_size.boundary,
        # lambda_weighting={"convective_theta": walls_sdf},
        criteria=Not(Eq(z, z0)),
    )
    domain.add_constraint(convective, "convective")

    vtk_obj = VTKFromFile(
        file_path=to_absolute_path(dir_path + "/temp_sol.vtu"),
        export_map={"Temperature": ["theta_star"], "Temperature_true": ["theta"]},
    )

    grid_inferencer = PointVTKInferencer(
        vtk_obj=vtk_obj,
        nodes=nodes,
        input_vtk_map={"x": "x", "y": "y", "z": "z"},
        output_names=["theta_star", "theta"],
        batch_size=1024,
    )
    domain.add_inferencer(grid_inferencer, "vtk_inf")

    # grid_validator = PointVTKValidator(
    #     vtk_obj=vtk_obj,
    #     nodes=nodes,
    #     input_vtk_map={"x": "x", "y": "y", "z": "z"},
    #     true_vtk_map={"theta": ["Temperature"]},
    #     requires_grad=False,
    #     batch_size=1024,
    # )
    # domain.add_validator(grid_validator, "vtk_val")

    # make solver
    slv = Solver(cfg, domain)

    # start solver
    slv.solve()


if __name__ == "__main__":
    run()
