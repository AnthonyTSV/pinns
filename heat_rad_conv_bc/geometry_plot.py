import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.path as mpath
import matplotlib as mpl
import numpy as np

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

plt.rcParams['text.usetex'] = True  
plt.rcParams['font.size'] = 16
plt.rcParams['font.family'] = 'Arial'

fig, ax = plt.subplots(figsize=(8, 4))

W = 4.0     # width of the main domain
H = 2.0     # height of the main domain
gap = 0.4   # thickness of the top "ambient" region
side = 0.4  # thickness of the hot regions on left/right

def curly_arrow(start, end, arr_size = 1, n = 5, col='gray', linew=1., width = 0.1):
    """
    https://stackoverflow.com/questions/45365158/matplotlib-wavy-arrow
    """
    xmin, ymin = start
    xmax, ymax = end
    dist = np.sqrt((xmin - xmax)**2 + (ymin - ymax)**2)
    n0 = dist / (2 * np.pi)
    
    x = np.linspace(0, dist, 151) + xmin
    y = width * np.sin(n * x / n0) + ymin
    line = plt.Line2D(x,y, color=col, lw=linew)
    
    del_x = xmax - xmin
    del_y = ymax - ymin
    ang = np.arctan2(del_y, del_x)
    
    line.set_transform(mpl.transforms.Affine2D().rotate_around(xmin, ymin, ang) + ax.transData)
    ax.add_line(line)

    verts = np.array([[0,1],[0,-1],[2,0],[0,1]]).astype(float) * arr_size
    verts[:,1] += ymax
    verts[:,0] += xmax
    path = mpath.Path(verts)
    patch = patches.PathPatch(path, fc=col, ec=col)

    patch.set_transform(mpl.transforms.Affine2D().rotate_around(xmax, ymax, ang) + ax.transData)
    return patch

# Left hot region
rect_left = patches.Rectangle(
    (-side, 0), side, H,
    facecolor='mistyrose',
    edgecolor='red',
    linewidth=1
)
ax.add_patch(rect_left)

# Right hot region
rect_right = patches.Rectangle(
    (W, 0), side, H,
    facecolor='mistyrose',
    edgecolor='red',
    linewidth=1
)
ax.add_patch(rect_right)

# Main domain region
rect_domain = patches.Rectangle(
    (0, 0), W, H,
    facecolor='peachpuff',
    edgecolor='none',
    linewidth=1
)
ax.add_patch(rect_domain)

# Top ambient region
rect_top = patches.Rectangle(
    (-gap, H), W+2*gap, gap,
    facecolor='lightblue',
    edgecolor='lightblue',
    linewidth=0
)
ax.add_patch(rect_top)

# Bottom boundary with a simple hatch
rect_bottom = patches.Rectangle(
    (0, -0.05), W, 0.05,
    facecolor='none',
    edgecolor='gray',
    hatch='///'
)
ax.add_patch(rect_bottom)

# Hot boundary labels
ax.text(0 + 0.1, H/2, r"$T_{\mathrm{hot}} = 1173\,[\mathrm{K}]$",
        color='red', rotation=90,
        ha='center', va='center')
ax.text(W - 0.1, H/2, r"$T_{\mathrm{hot}} = 1173\,[\mathrm{K}]$",
        color='red', rotation=90,
        ha='center', va='center')

# Top boundary label
ax.text(W/2, H+2*gap, r"$T_{\mathrm{amb}} = 323\,[\mathrm{K}]$",
        color='blue', ha='center', va='center')

# Domain label
ax.text(W/2, H/2, r"$\Omega$", color='black', fontsize=16,
        ha='center', va='center')

# Dimension arrows for W and H
# Horizontal arrow for width
ax.annotate("",
            xy=(0, -0.15), xytext=(W, -0.15),
            arrowprops=dict(arrowstyle="<->", lw=1.2, color='black'))
ax.text(W/2, -0.17, r"$W = 2\,[m]$", ha='center', va='top')

# Vertical arrow for height
ax.annotate("",
            xy=(W+side+0.2, 0), xytext=(W+side+0.2, H),
            
            arrowprops=dict(arrowstyle="<->", lw=1.2, color='black'))
ax.text(W+side+0.25, H/2, r"$H = 1\,[m]$", ha='left', va='center')

x_positions = np.linspace(0, W - side, 5)
for x in x_positions:
    ax.annotate("",
                xy=(x, H+gap), xytext=(x, H),
                arrowprops=dict(arrowstyle="->", color='orange', lw=1.5))
    ax.add_patch(curly_arrow((x+side, H), (x+side, H+gap/2), col='red', width=0.02, n=2, arr_size=0.03))

# Labels for q_rad and q_conv near the top right
ax.text(W+0.1*side, H+0.2, r"$q_{\mathrm{rad}}$", color='orange', ha='left', va='bottom')
ax.text(W+0.1*side, H+0.02, r"$q_{\mathrm{conv}}$", color='red', ha='left', va='bottom')

ax.set_xlim(-side-0.5, W+side+1.0)
ax.set_ylim(-0.2, H+gap+0.5)
ax.set_aspect('equal')
ax.axis('off')

plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, 'figures/rad_conv_geometry.pdf'), dpi=300)
