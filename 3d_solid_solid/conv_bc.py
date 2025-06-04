from sympy import Symbol, Function, Number

from physicsnemo.sym.eq.pde import PDE
from physicsnemo.sym.node import Node

class ConvectiveBC(PDE):

    name = "ConvectiveBC"

    def __init__(self, T, kappa, h, T_ext, dim = 3 , time = False, non_dim = False):
        self.T = T
        self.dim = dim
        self.time = time

        x, y, z = Symbol("x"), Symbol("y"), Symbol("z")
        normal_x = Symbol("normal_x")
        normal_y = Symbol("normal_y")
        normal_z = Symbol("normal_z")

        t = Symbol("t")

        input_variables = {"x": x, "y": y, "z": z, "t": t}
        if self.dim == 1:
            input_variables.pop("y")
            input_variables.pop("z")
        elif self.dim == 2:
            input_variables.pop("z")
        if not self.time:
            input_variables.pop("t")

        T = Function(T)(*input_variables)
        self.equations = {}
        if not non_dim:
            self.equations["convective_" + self.T] = (
                kappa * (normal_x * T.diff(x)  + normal_y * T.diff(y) + normal_z * T.diff(z)) + h * (T - T_ext)
            )
        else:
            self.equations["convective_" + self.T] = (
                (normal_x * T.diff(x)  + normal_y * T.diff(y) + normal_z * T.diff(z)) + h * T
            )