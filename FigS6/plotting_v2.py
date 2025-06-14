import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator, tick_params
fig, ax = plt.subplots()
plt.rcParams['font.family'] = 'DeJavu Serif'
plt.rcParams["font.family"] = "Helvetica"

# ==============================================================================================
#                                       Global Parameters     
# ==============================================================================================
lw = 5.0
legendsize = 48         # size for legend
font_legend = {'family':'Times New Roman', 'weight': 'roman', 'size': 23}
font_legends = {'family':'Times New Roman', 'weight': 'roman', 'size': 19}
# axis label size
lsize = 30             
txtsize = 32
# tick length
lmajortick = 15
lminortick = 5
legend_x, legend_y = - 0.12, 1.03
transparency = .4

unitlen = 7
fig = plt.figure(figsize=(1.2 * unitlen, 1.0 * unitlen), dpi = 128)

# ==============================================================

conv = 27.211397                            # 1 a.u. = 27.211397 eV
au_to_K = 3.1577464e+05                     # 1 au = 3.1577464e+05 K
fs2au = 41.341374575751                     # 1 fs = 41.341374575751 a.u.
mm2au = 18897261.257078                     # 1 mm = 18897261.257078 a.u.
au2ans = 0.529177                           # 1 a.u. = 0.529177 ans
c2au = 137.0359895                          # speed of light
um_ps2au = (mm2au / 1000) / (1000 * fs2au)  # um / ps in a.u.

L = 0.04 * mm2au                                                            # box lenght
dk = 2 * np.pi / L                                                          # k-space unit length
dk0 = dk * (40 / (2 * np.pi))                                                # convert to micron^-1

# ==============================================================================================
#                           Fig 2a 
# ==============================================================================================

x = np.loadtxt("kpar_lambda=12meV.txt", dtype = float) / dk0
y = np.loadtxt("wc_lambda=12meV.txt", dtype = float) * conv
z = np.loadtxt("vg_LP_lambda=12meV.txt", dtype = float).transpose() / um_ps2au

x, y = np.meshgrid(x, y)
plt.pcolormesh(x, y, z, cmap = 'RdYlBu', shading = 'gouraud', vmin = z.min(), vmax = 0) # binary_r magma summer 



cbar = plt.colorbar(ticks = np.linspace(-9, 0, 10))
cbar.ax.tick_params(labelsize = 20, which = 'both', direction = 'out')
cbar.ax.get_yaxis().labelpad = 20
cbar.set_label(r'$\Delta v_{g,-}~(\mu \mathrm{m / ps})$', fontdict = font_legend)

# ==============================================================================================
# RHS y-axis
ax = plt.gca()
ax.plot([0, 7], [1.96, 1.96], '--', linewidth = lw, color = 'navy', label = 'Exciton')
ax.text(3, 1.9, r'$\hbar \omega_0 = 1.96$ eV', fontsize = 24, color = 'navy')#, font = 'Helvetica') # 3
x_major_locator = MultipleLocator(2)
x_minor_locator = MultipleLocator(1)
y_major_locator = MultipleLocator(0.1)
y_minor_locator = MultipleLocator(0.05)
ax.xaxis.set_major_locator(x_major_locator)
ax.xaxis.set_minor_locator(x_minor_locator)
ax.yaxis.set_major_locator(y_major_locator)
ax.yaxis.set_minor_locator(y_minor_locator)
ax.tick_params(which = 'major', length = 15, pad = 10)
ax.tick_params(which = 'minor', length = 5)

x1_label = ax.get_xticklabels()
[x1_label_temp.set_fontname('Times New Roman') for x1_label_temp in x1_label]
y1_label = ax.get_yticklabels()
[y1_label_temp.set_fontname('Times New Roman') for y1_label_temp in y1_label]

plt.tick_params(which = 'both', direction = 'in', labelsize = 30)
plt.xlim(0, 7)
plt.ylim(1.55, 2.1)

# RHS y-axis
ax2 = ax.twinx()
ax2.yaxis.set_major_locator(y_major_locator)
ax2.yaxis.set_minor_locator(y_minor_locator)
ax2.tick_params(which = 'major', length = 15)
ax2.tick_params(which = 'minor', length = 5)
ax2.axes.yaxis.set_ticklabels([])

plt.tick_params(which = 'both', direction = 'in')
plt.ylim(1.55, 2.1)

ax.set_xlabel(r'$k_\parallel$ ($\mu$m$^{-1}$)', size = 32)
ax.set_ylabel(r'$\omega_\mathrm{c}$ (eV)', size = 32)
# ax.legend()



plt.savefig("Fig_S6.pdf", bbox_inches='tight')