import os
import numpy as np
from sympy import Symbol, sin, Eq, Or, Abs

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
from physicsnemo.sym.eq.pdes.navier_stokes import GradNormal

from physicsnemo.sym.utils.io.vtk import VTKFromFile, VTKUniformGrid
from physicsnemo.sym.domain.validator import PointVTKValidator
from physicsnemo.sym.domain.inferencer import PointVTKInferencer
from physicsnemo.sym.utils.io.vtk import var_to_polyvtk
from physicsnemo.sym.utils.io import InferencerPlotter, ValidatorPlotter
from physicsnemo.models.layers.activations import get_activation

from physicsnemo.sym.models.fully_connected import FullyConnectedArch
from conv_bc import ConvectiveBC
import matplotlib.pyplot as plt
import scienceplots

plt.style.use("science")

dir_path = os.path.dirname(os.path.realpath(__file__))

os.environ["HYDRA_FULL_ERROR"] = "1"

@physicsnemo.sym.main(config_path="conf", config_name="config")
def run(cfg: PhysicsNeMoConfig) -> None:
    cfg.network_dir = dir_path+'/outputs'
    cfg.initialization_network_dir = dir_path+'/outputs'
    
    plotter = ValidatorPlotter()

    w_sym = 2 # m
    h_sym = 1 # m
    L = h_sym
    kappa_dim = 3 # W/mK
    h_dim = 50 # W/m^2K
    epsilon = 0.7 # emissivity
    sigma = 5.67e-8 # Stefan-Boltzmann constant
    T_ext = 323 # K
    T_hot = 1173 # K
    delta_T = T_hot - T_ext

    # Biot number
    Bi = h_dim * L * (4*epsilon*sigma*T_ext**3) / kappa_dim
    kappa_dimless = 1.0
    T_ext_dimless = 0.0
    T_hot_dimless = 1.0

    # make list of nodes to unroll graph on
    heat_eq = Diffusion(T="theta", D=kappa_dimless, dim=2, time=False)
    grad_u = GradNormal("theta", dim=2, time=False)
    conv_u = ConvectiveBC("theta", kappa_dimless, Bi, T_ext_dimless, dim=2, time=False)
    input_keys = [Key("x"), Key("y")]
    T_phys_node = Node.from_sympy(
        eq=Symbol("theta") * delta_T + T_ext,
        out_name="theta_phys",
    )
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
    nodes = (
        heat_eq.make_nodes() + 
        grad_u.make_nodes() + 
        conv_u.make_nodes() + 
        [heat_net.make_node(name="heat_network")] +
        [T_phys_node]
    )

    # make geometry
    x, y = Symbol("x"), Symbol("y")
    geo = Rectangle(
        point_1=(-w_sym / 2, -h_sym / 2),
        point_2=(w_sym / 2, h_sym / 2),
    )

    # make domain
    domain = Domain()

    # boundary condition
    top_bc = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=geo,
        outvar={"convective_theta": T_ext_dimless},
        batch_size=cfg.batch_size.wall,
        lambda_weighting={"convective_theta": 1.0 - Abs(x)},
        criteria=Eq(y, h_sym / 2),
    )
    domain.add_constraint(top_bc, "left_bc")

    left_right_bc = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=geo,
        outvar={"theta": T_hot_dimless},
        batch_size=cfg.batch_size.wall,
        lambda_weighting={"theta": 5},
        criteria=Or(Eq(x, -w_sym / 2), Eq(x, w_sym / 2)),
    )
    domain.add_constraint(left_right_bc, "left_right_bc")

    # bottom wall
    bottom_bc = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=geo,
        outvar={"normal_gradient_theta": 0},
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
        file_path=to_absolute_path(dir_path + "/fem_data/temperature_solution.vtu"),
        export_map={"Temperature": ["theta_phys"]},
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
        true_vtk_map={"theta_phys": ["Temperature"]},
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