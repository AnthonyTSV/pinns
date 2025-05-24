from sympy import Symbol, Function, Number
from modulus.sym.eq.pde import PDE

class HeatEquation(PDE):
    name = "HeatEquation"

    def __init__(self, kappa: float = 1.0, time=False):
        x = Symbol("x")
        y = Symbol("y")
        t = Symbol("t")

        input_variables = {"x": x, "y": y}

        u = Function("u")(*input_variables)

        kappa = Number(kappa)

        self.equations = {}
        self.equations["heat_equation"] = (
            u.diff(t) - (kappa * u.diff(x)).diff(x) - (kappa * u.diff(y)).diff(y)
        )

