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
from conv_bc import ConvectiveBC
from modulus.sym.eq.pdes.basic import GradNormal

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

    L = 1 # m
    kappa_dim = 52 # W/mK
    h_dim = 750 # W/m^2K
    T_ext = 0.0

    Bi = h_dim * L / kappa_dim
    kappa_dimless = 1.0
    T_ext_dimless = 0.0

    # make list of nodes to unroll graph on
    heat_eq = HeatEquation(kappa=kappa_dimless, time=False)
    grad_u = GradNormal("u", dim=2, time=False)
    conv_u = ConvectiveBC("u", kappa_dimless, Bi, T_ext_dimless, dim=2, time=False)
    input_keys = [Key("x"), Key("y")]
    T_phys_node = Node.from_sympy(
        eq=Symbol("u") * 100,
        out_name="u_phys",
    )
    heat_net = instantiate_arch(
        input_keys=input_keys,
        output_keys=[Key("u")],
        cfg=cfg.arch.fully_connected,
    )
    nodes = (
        heat_eq.make_nodes() + 
        grad_u.make_nodes() + 
        conv_u.make_nodes() + 
        [heat_net.make_node(name="heat_network")] +
        [T_phys_node]
    )

    # make geometry
    w_sym = 1 # m
    h_sym = 1 # m
    x, y = Symbol("x"), Symbol("y")
    T_bottom = 100
    geo = Rectangle(
        point_1=(-w_sym / 2, -h_sym / 2),
        point_2=(w_sym / 2, h_sym / 2),
    )

    # make domain
    domain = Domain()

    # boundary condition
    left_bc = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=geo,
        outvar={"normal_gradient_u": 0},
        batch_size=cfg.batch_size.wall,
        criteria=Eq(x, -w_sym / 2),
    )
    domain.add_constraint(left_bc, "left_bc")

    top_right_bc = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=geo,
        outvar={"convective_u": 0},
        batch_size=cfg.batch_size.wall,
        criteria=Or(Eq(y, h_sym / 2), Eq(x, w_sym / 2)),
    )
    domain.add_constraint(top_right_bc, "top_right_bc")

    # bottom wall
    bottom_bc = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=geo,
        outvar={"u": 1},
        batch_size=cfg.batch_size.wall,
        criteria=Eq(y, -h_sym / 2),
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
    )
    domain.add_constraint(interior, "interior")

    vtk_obj = VTKUniformGrid(
        bounds=[[-w_sym / 2, w_sym / 2], [-h_sym / 2, h_sym / 2]],
        npoints=[128, 128],
        export_map={"u_phys": ["u_phys"], "u": ["u"]},
    )
    grid_inference = PointVTKInferencer(
        vtk_obj=vtk_obj,
        nodes=nodes,
        input_vtk_map={"x": "x", "y": "y"},
        output_names=["u_phys", "u"],
        requires_grad=False,
        batch_size=1024,
    )
    domain.add_inferencer(grid_inference, "vtk_inf")

    # make solver
    slv = Solver(cfg, domain)

    # start solver
    slv.solve()


if __name__ == "__main__":
    run()