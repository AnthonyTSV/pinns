import matplotlib.backends
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.patches import Rectangle
from matplotlib.gridspec import GridSpec
import os
import scienceplots
matplotlib.style.use("science")
matplotlib.use('TkAgg')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

L_base = 0.65
W_base = 0.8625
H_base = 0.05

L_fin  = 0.65
W_fin  = 0.05
H_fin  = 0.8625

L_src = 0.25
W_src = 0.25

src_x0 = (L_base - L_src) / 2
src_y0 = (W_base - W_src) / 2

# Helper – draw double‑headed arrow with text
def dim_arrow(ax, start, end, text, text_offset=(0,0)):
    ax.annotate(
        "",
        xy=end, xytext=start,
        arrowprops=dict(arrowstyle="<->", shrinkA=0, shrinkB=0),
        annotation_clip=False
    )
    mid_x = (start[0] + end[0]) / 2 + text_offset[0]
    mid_y = (start[1] + end[1]) / 2 + text_offset[1]
    ax.text(mid_x, mid_y, text, ha='center', va='center')

fig = plt.figure(figsize=(8, 6))
gs = GridSpec(nrows=2, ncols=2, figure=fig,
              width_ratios=[1, 1],    # left column same width as right column
              height_ratios=[1, 1])   # each row same height

# Left column, spanning both rows
ax_img = fig.add_subplot(gs[:, 0])   # all rows in col 0

# Top‐right: XY top view
ax_xy  = fig.add_subplot(gs[0, 1])

# Bottom‐right: YZ side view (with 3 fins)
ax_yz  = fig.add_subplot(gs[1, 1])

png_path = BASE_DIR + "/heat_sink_figures/heat_sink.png"
img = plt.imread(png_path)
ax_img.imshow(img)
ax_img.axis("off")

# XY – top view
ax_xy.add_patch(Rectangle((0, 0), L_base, W_base, fill=True, color='lightgray', alpha=0.8))
ax_xy.add_patch(Rectangle((src_x0, src_y0), L_src, W_src, fill=True, color='red', alpha=0.8))
# Dimension arrows
dim_arrow(ax_xy, (0, W_base + 0.05), (L_base, W_base + 0.05), f"{L_base}", text_offset=(0, 0.05))
dim_arrow(ax_xy, (-0.05, 0), (-0.05, W_base), f"{W_base}", text_offset=(-0.13, 0))
# Heat source arrow
dim_arrow(ax_xy, (src_x0 - 0.01, src_y0 - 0.02), (src_x0 + L_src + 0.01, src_y0 - 0.02), f"{L_src}", text_offset=(0, -0.07))
dim_arrow(ax_xy, (src_x0 - 0.02, src_y0 - 0.01), (src_x0 - 0.02, src_y0 + W_src + 0.01), f"{W_src}", text_offset=(-0.1, 0))
# Labels / aesthetic
ax_xy.set_xlim(-0.15, L_base + 0.15)
ax_xy.set_ylim(-0.15, W_base + 0.15)
ax_xy.set_aspect('equal')
ax_xy.axis('off')

dy = W_base
nfins, fin_w, fin_h = 5, W_fin, H_fin
gap = (dy - nfins * fin_w) / (nfins - 1) if nfins > 1 else 0.0

# YZ – side view
# Base plate
ax_yz.add_patch(Rectangle((0, 0), W_base, H_base, fill=True, color='lightgray', alpha=0.8, edgecolor=None))
# Fins
for i in range(nfins):
    y_offset = i * (fin_w + gap)
    ax_yz.add_patch(Rectangle((y_offset, W_fin), W_fin, H_fin, fill=True, color='lightgray', alpha=0.8, edgecolor=None))
# Heat source
ax_yz.add_patch(Rectangle((W_base / 2 - L_src / 2, 0), L_src, 0.01, fill=True, color='red', alpha=0.8, edgecolor=None))
# Dimension arrows
dim_arrow(ax_yz, (0, -0.05), (W_base, -0.05), f"{W_base}", text_offset=(0, -0.05))
dim_arrow(ax_yz, (-0.01, H_base), (-0.01, H_base + H_fin), f"{H_fin}", text_offset=(-0.12, 0))
ax_yz.text(0.02, H_base + H_fin + 0.05, f"{W_fin}", ha='center', va='center')
ax_yz.text
ax_yz.set_xlim(-0.15, W_base + 0.15)
ax_yz.set_ylim(-0.05, H_base + H_fin + 0.1)
ax_yz.set_aspect('equal')
ax_yz.axis('off')

plt.savefig(os.path.join(BASE_DIR, "heat_sink_figures/heat_sink_geometry.pdf"), dpi=300, bbox_inches="tight")
