import numpy as np
import matplotlib.pyplot as plt
from sympy import lambdify
from sympy import Symbol, tanh

x, y, z = Symbol("x"), Symbol("y"), Symbol("z")

x0, y0, z0 = -0.75, -0.50, -0.4375
dx, dy, dz = 0.2, 0.2, 0.01

source_origin = ((x0 + x0 + dx) / 2, (y0 + y0 + dy) / 2, z0)
source_dim = (0.25, 0.25, 0)
source_grad = 100

xc, yc = (x0 + dx/2), (y0 + dy/2)        # plate centre
wx, wy  = 0.10, 0.10                     # desired patch size  (≤ dx, dy)

xl, xr = xc - wx/2, xc + wx/2            # left / right edges
yl, yr = yc - wy/2, yc + wy/2            # bottom / top edges

# --- smoothed indicator -------------------------------------------------
a = 60.0                                 # 'sharpen_tanh'

step_lx = 0.5*(tanh(a*(x - xl)) + 1.0)   # switches ON at  x > xl
step_rx = 0.5*(tanh(a*(xr - x)) + 1.0)   # switches OFF at x > xr
step_ly = 0.5*(tanh(a*(y - yl)) + 1.0)
step_ry = 0.5*(tanh(a*(yr - y)) + 1.0)

indicator = step_lx * step_rx * step_ly * step_ry
gradient_normal = source_grad * indicator

grad_fn = lambdify((x, y, z), gradient_normal, modules="numpy")

nx, ny = 400, 400
xs = np.linspace(x0, x0 + dx, nx)
ys = np.linspace(y0, y0 + dy, ny)
X, Y = np.meshgrid(xs, ys)
Z = np.full_like(X, z0)

# 3.  Evaluate and plot
G = grad_fn(X, Y, Z)

plt.figure(figsize=(6, 5))
pcm = plt.contourf(X, Y, G, 51)        # filled contours
plt.colorbar(pcm, label=r"$\partial\theta/\partial n$  (W m$^{-2}$ K$^{-1}$)")
plt.xlabel("x  [m]");  plt.ylabel("y  [m]")
plt.title("Prescribed normal gradient on the bottom surface")
plt.gca().set_aspect("equal")
plt.tight_layout()
plt.show()
