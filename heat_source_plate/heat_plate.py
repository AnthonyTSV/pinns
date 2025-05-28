import os
import numpy as np
from sympy import Symbol, sin, Eq, Or, Abs, tanh

import physicsnemo.sym
from physicsnemo.sym.hydra import instantiate_arch, PhysicsNeMoConfig, to_absolute_path
from physicsnemo.sym.solver import Solver
from physicsnemo.sym.domain import Domain
from physicsnemo.sym.geometry.primitives_2d import Rectangle, Line
from physicsnemo.sym.domain.constraint import (
    PointwiseBoundaryConstraint,
    PointwiseInteriorConstraint,
)
from physicsnemo.sym.geometry import Parameterization
from physicsnemo.sym.domain.validator import PointwiseValidator
from physicsnemo.sym.key import Key
from physicsnemo.sym.node import Node
from physicsnemo.sym.eq.pdes.diffusion import Diffusion

from physicsnemo.sym.utils.io.vtk import VTKFromFile, VTKUniformGrid
from physicsnemo.sym.domain.validator import PointVTKValidator
from physicsnemo.sym.domain.inferencer import PointVTKInferencer
from physicsnemo.sym.utils.io.vtk import var_to_polyvtk
from physicsnemo.sym.utils.io import InferencerPlotter, ValidatorPlotter
from physicsnemo.sym.eq.pdes.navier_stokes import NavierStokes, GradNormal
from conv_bc import ConvectiveBC
from physicsnemo.models.layers.activations import get_activation

from physicsnemo.sym.models.fully_connected import FullyConnectedArch
import matplotlib.pyplot as plt

dir_path = os.path.dirname(os.path.realpath(__file__))

os.environ["HYDRA_FULL_ERROR"] = "1"


@physicsnemo.sym.main(config_path="conf", config_name="config")
def run(cfg: PhysicsNeMoConfig) -> None:
    plotter = ValidatorPlotter()
    # make list of nodes to unroll graph on
    kappa = 3
    h_conv = 50
    ambient_temp = 30
    heat_eq = Diffusion(T="theta", D=kappa, dim=2, Q=1.0, time=False)
    grad_theta = GradNormal("theta", dim=2, time=False)
    conv_theta = ConvectiveBC(
        T="theta", kappa=kappa, h=h_conv, T_ext=ambient_temp, dim=2, time=False
    )
    x, y = Symbol("x"), Symbol("y")
    input_keys = [Key("x"), Key("y")]
    w_sym = 1
    h_sym = 1
    
    activation_function = cfg.custom.activation
    interior_points = cfg.batch_size.interior
    layer_size = cfg.custom.layer_size
    num_layers = cfg.custom.num_layers
    heat_net = FullyConnectedArch(
        input_keys=input_keys,
        output_keys=[Key("theta")],
        layer_size=layer_size,
        nr_layers=num_layers,
        activation_fn=get_activation(activation_function)
    )
    cfg.network_dir = dir_path + f"/outputs/{activation_function}_interior_{interior_points}_arch_{layer_size}x{num_layers}"
    cfg.initialization_network_dir = dir_path + f"/outputs/{activation_function}_interior_{interior_points}_arch_{layer_size}x{num_layers}"
    if cfg.custom.parametric:
        cfg.network_dir = dir_path + f"/outputs/parametric"
        cfg.initialization_network_dir = dir_path + f"/outputs/parametric"
    nodes = heat_eq.make_nodes() + [heat_net.make_node(name="heat_network")] + grad_theta.make_nodes() + conv_theta.make_nodes()

    # add constraints to solver, make geometry
    
    geo = Rectangle(
        point_1=(-w_sym / 2, -h_sym / 2),
        point_2=(w_sym / 2, h_sym / 2),
    )

    # make domain
    domain = Domain()

    # boundary condition
    not_bottom = Or(
        Eq(y, h_sym / 2),  # top wall
        Eq(x, -w_sym / 2),  # left wall
        Eq(x, w_sym / 2),   # right wall
    )
    not_bottom_bc = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=geo,
        outvar={"convective_theta": 0},
        batch_size=cfg.batch_size.wall,
        criteria=not_bottom,
    )
    domain.add_constraint(not_bottom_bc, "not_bottom_bc")

    source_grad = 100
    x0 = 0
    xc = (x0)
    wx  = 0.50

    xl, xr = xc - wx/2, xc + wx/2

    a = 60.0

    step_lx = 0.5*(tanh(a*(x - xl)) + 1.0)
    step_rx = 0.5*(tanh(a*(xr - x)) + 1.0)

    indicator = step_lx * step_rx
    gradient_normal = source_grad * indicator
    # bottom wall
    bottom_bc = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=geo,
        outvar={"normal_gradient_theta": gradient_normal},
        batch_size=cfg.batch_size.wall,
        criteria=Eq(y, -h_sym / 2),
    )
    domain.add_constraint(bottom_bc, "bottom_bc")

    # interior
    interior = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=geo,
        outvar={"diffusion_theta": 0},
        batch_size=cfg.batch_size.interior,
        lambda_weighting={
            "diffusion_theta": Symbol("sdf"),
        },
    )
    domain.add_constraint(interior, "interior")
    
    vtk_obj = VTKFromFile(
        file_path=to_absolute_path(dir_path + "/temperature_solution.vtu"),
        export_map={"Temperature": ["theta"]},
    )
    # center the grid
    points = vtk_obj.get_points()
    points[:, 0] -= w_sym / 2
    points[:, 1] -= h_sym / 2
    vtk_obj.set_points(points)
    validator = PointVTKValidator(
        vtk_obj=vtk_obj,
        nodes=nodes,
        input_vtk_map={"x": ["x"], "y": ["y"]},
        true_vtk_map={"theta": ["Temperature"]},
        requires_grad=False,
        batch_size=128 * 128,
        plotter=plotter,
    )
    domain.add_validator(validator, "vtk_val")

    # make solver
    slv = Solver(cfg, domain)

    # start solver
    slv.solve()


if __name__ == "__main__":
    run()
