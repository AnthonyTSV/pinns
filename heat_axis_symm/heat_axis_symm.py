import os
import numpy as np
from sympy import Symbol, sin, Eq, Or, Abs

import modulus.sym
from modulus.sym.hydra import instantiate_arch, ModulusConfig, to_absolute_path
from modulus.sym.solver import Solver
from modulus.sym.domain import Domain
from modulus.sym.geometry.primitives_2d import Rectangle, Line
from modulus.sym.domain.constraint import (
    PointwiseBoundaryConstraint,
    PointwiseInteriorConstraint,
)
from modulus.sym.geometry import Parameterization
from modulus.sym.domain.validator import PointwiseValidator
from modulus.sym.key import Key
from modulus.sym.node import Node
from heat_equation import HeatEquation
from heat_flux import HeatFlux
from modulus.sym.eq.pdes.basic import GradNormal

from modulus.sym.utils.io.vtk import VTKFromFile, VTKUniformGrid
from modulus.sym.domain.validator import PointVTKValidator
from modulus.sym.domain.inferencer import PointVTKInferencer
from modulus.sym.utils.io.vtk import var_to_polyvtk

from modulus.sym.models.fully_connected import FullyConnectedArch
from modulus.sym.utils.io import InferencerPlotter, ValidatorPlotter
from modulus.sym.domain.inferencer import PointwiseInferencer
from modulus.models.layers.activations import get_activation

dir_path = os.path.dirname(os.path.realpath(__file__))

os.environ["HYDRA_FULL_ERROR"] = "1"

@modulus.sym.main(config_path="conf", config_name="config")
def run(cfg: ModulusConfig) -> None:
    cfg.network_dir = dir_path+'/outputs'
    cfg.initialization_network_dir = dir_path+'/outputs'
    
    plotter = ValidatorPlotter()

    r_in = 0.02
    r_out = 0.1
    width = r_out - r_in
    height = 0.14
    delta_p = 0.06

    w_sym = width # m
    h_sym = height # m
    L = h_sym
    kappa_dim = 52 # W/mK
    q_dim = 5e2 # W/m^2
    hight_t = 600
    low_t = 273.15

    # make list of nodes to unroll graph on
    heat_eq = HeatEquation(kappa=kappa_dim, time=False)
    grad_u = GradNormal("u", dim=2, time=False)
    input_keys = [Key("x"), Key("y")]
    heat_net = instantiate_arch(
        input_keys=input_keys,
        output_keys=[Key("u")],
        cfg=cfg.arch.fully_connected,
    )
    heat_net = FullyConnectedArch(
        input_keys=input_keys,
        output_keys=[Key("u")],
        layer_size=512,
        nr_layers=6,
        # activation_fn=get_activation("gelu")
    )
    nodes = (
        heat_eq.make_nodes() + 
        grad_u.make_nodes() +
        # heat_flux.make_nodes() +
        [heat_net.make_node(name="heat_network")]
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
    flux_bc = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=geo,
        outvar={"normal_gradient_u": q_dim},
        batch_size=cfg.batch_size.wall,
        # lambda_weighting={"heat_flux_u": 1.0 - Abs(x)},
        criteria=(Eq(x, -w_sym/2) & (y > -delta_p/2) & (y < delta_p/2)),
    )
    domain.add_constraint(flux_bc, "left_bc")

    neumann_bc = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=geo,
        outvar={"normal_gradient_u": 0},
        batch_size=cfg.batch_size.wall,
        # lambda_weighting={"heat_flux_u": 1.0 - Abs(x)},
        criteria=(
            Eq(x, -w_sym / 2)
            & (
                ((y < -delta_p/2) & (y > -h_sym/2)) |
                ((y > delta_p/2) & (y < h_sym/2))
            )
        ),
    )
    domain.add_constraint(neumann_bc, "neumann_bc")

    walls_criteria = Or(
        Eq(y, h_sym / 2), 
        Eq(x, w_sym / 2),
        Eq(y, -h_sym / 2),
    )

    walls = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=geo,
        outvar={"u": low_t},
        batch_size=cfg.batch_size.wall,
        # lambda_weighting={"u": 1},
        criteria=walls_criteria,
    )
    domain.add_constraint(walls, "walls")

    # interior
    interior = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=geo,
        outvar={"heat_equation": 0},
        batch_size=cfg.batch_size.interior,
        lambda_weighting={
            "heat_equation": Symbol("sdf"),
        },
    )
    domain.add_constraint(interior, "interior")

    vtk_obj = VTKFromFile(
        to_absolute_path("./heat_axis_symm/fem_data/temperature_solution.vtu"),
        export_map={"u": ["Temperature"]},
    )

    grid_inference = PointVTKInferencer(
        vtk_obj=vtk_obj,
        nodes=nodes,
        input_vtk_map={"x": "x", "y": "y"},
        output_names=["u"],
        requires_grad=False,
        batch_size=1024,
        # plotter=plotter,
    )
    domain.add_inferencer(grid_inference, "vtk_inf")

    points = vtk_obj.get_points()
    points[:, 0] += -width / 2  # center data
    points[:, 1] += -height / 2  # center data
    vtk_obj.set_points(points)

    vtk_validator = PointVTKValidator(
        nodes=nodes,
        vtk_obj=vtk_obj,
        input_vtk_map={"x": "x", "y": "y"},
        true_vtk_map={"u": ["Temperature"]},
        requires_grad=False,
        batch_size=1024,
        plotter=plotter,
    )
    domain.add_validator(vtk_validator)

    # make solver
    slv = Solver(cfg, domain)

    # start solver
    slv.solve()


if __name__ == "__main__":
    run()