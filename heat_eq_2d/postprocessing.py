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

from modulus.sym.utils.io.vtk import VTKFromFile, VTKUniformGrid
from modulus.sym.domain.validator import PointVTKValidator
from modulus.sym.domain.inferencer import PointVTKInferencer
from modulus.sym.utils.io.vtk import var_to_polyvtk

from modulus.sym.models.fully_connected import FullyConnectedArch

dir_path = os.path.dirname(os.path.realpath(__file__))

os.environ["HYDRA_FULL_ERROR"] = "1"

@modulus.sym.main(config_path="conf", config_name="config")
def run(cfg: ModulusConfig) -> None:
    cfg.network_dir = dir_path+'/outputs'
    cfg.initialization_network_dir = dir_path+'/outputs'
    # make list of nodes to unroll graph on
    heat_eq = HeatEquation(kappa=1.0, time=False)
    input_keys = [Key("x"), Key("y"), Key("w"), Key("h"), Key("T_left")]
    heat_net = instantiate_arch(
        input_keys=input_keys,
        output_keys=[Key("u")],
        cfg=cfg.arch.fully_connected,
    )
    nodes = heat_eq.make_nodes() + [heat_net.make_node(name="heat_network")]

    # add constraints to solver
    # make geometry
    w_sym = Symbol("w")
    h_sym = Symbol("h")
    x, y = Symbol("x"), Symbol("y")
    T_left = Symbol("T_left")
    param_ranges = {
        "w": (0.1, 1.0),
        "h": (0.1, 1.0),
        "T_left": (5, 50),
    }
    parameterization = Parameterization(param_ranges)
    geo = Rectangle(
        point_1=(-w_sym / 2, -h_sym / 2),
        point_2=(w_sym / 2, h_sym / 2),
        parameterization=parameterization
    )

    # make domain
    domain = Domain()

    # boundary condition
    left_bc = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=geo,
        outvar={"u": T_left},
        batch_size=cfg.batch_size.left_wall,
        # lambda_weighting={"u": 25 - 50 * Abs(y)},
        criteria=Eq(x, -w_sym / 2),
        parameterization=parameterization
    )
    domain.add_constraint(left_bc, "left_bc")

    top_bc = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=geo,
        outvar={"u": 5},
        batch_size=cfg.batch_size.wall,
        criteria=Eq(y, h_sym / 2),
        parameterization=parameterization
    )
    domain.add_constraint(top_bc, "top_bc")

    # right wall
    right_bc = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=geo,
        outvar={"u": 5},
        batch_size=cfg.batch_size.wall,
        criteria=Eq(x, w_sym / 2),
        parameterization=parameterization
    )
    domain.add_constraint(right_bc, "right_bc")

    # bottom wall
    bottom_bc = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=geo,
        outvar={"u": 5},
        batch_size=cfg.batch_size.wall,
        criteria=Eq(y, -h_sym / 2),
        parameterization=parameterization
    )
    domain.add_constraint(bottom_bc, "bottom_bc")

    # interior
    interior = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=geo,
        outvar={"heat_equation": 0},
        batch_size=cfg.batch_size.interior,
        lambda_weighting={
            "heat_equation": Symbol("sdf"),
        },
        parameterization=parameterization
    )
    domain.add_constraint(interior, "interior")

    w_eval = 1
    h_eval = 1
    t_left = 50

    # Create 2D uniform grid of shape (128, 128)
    vtk_obj = VTKUniformGrid(
        bounds=[[-w_eval / 2, w_eval / 2], [-h_eval / 2, h_eval / 2]],
        npoints=[128, 128],
        export_map={"U": ["u"]},  # rename "u" -> "U" in the output file
    )

    inferencer = PointVTKInferencer(
        vtk_obj=vtk_obj,
        nodes=nodes,
        input_vtk_map={"x": ["x"], "y": ["y"]},
        output_names=["u"],
        invar={
            "w": np.full([128**2, 1], w_eval),
            "h": np.full([128**2, 1], h_eval),
            "T_left": np.full([128**2, 1], t_left)
        },  # Must match shape of x,y
        batch_size=128 * 128,
        requires_grad=False,
        log_iter=False,
    )

    domain.add_inferencer(inferencer, "postproc_inference")

    # make solver
    slv = Solver(cfg, domain)

    # start solver
    slv.eval()


if __name__ == "__main__":
    run()