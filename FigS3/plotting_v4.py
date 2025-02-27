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
fig = plt.figure(figsize=(2.5 * unitlen, 1.0 * unitlen), dpi = 128)
plt.subplots_adjust(wspace = 0.32)

# ==============================================================

conv = 27.211397                            # 1 a.u. = 27.211397 eV
au_to_K = 3.1577464e+05                     # 1 au = 3.1577464e+05 K
fs2au = 41.341374575751                     # 1 fs = 41.341374575751 a.u.
mm2au = 18897261.257078                     # 1 mm = 18897261.257078 a.u.
au2ans = 0.529177                           # 1 a.u. = 0.529177 ans
c2au = 137.0359895                          # speed of light
um_ps2au = (mm2au / 1000) / (1000 * fs2au)  # um / ps in a.u.

def wk(wc, k):
    return np.sqrt(wc**2 + c2au**2 * k**2)

def omega_LP(wc, k):
    theta = np.arctan(k / (wc / c2au))
    return 0.5 * (w0 + wk(wc, k)) - 0.5 * np.sqrt((wk(wc, k) - w0)**2 + 4 * Omega_R**2 * (wk(wc, k) / wc) * np.cos(theta)**2)

def omega_UP(wc, k):
    theta = np.arctan(k / (wc / c2au))
    return 0.5 * (w0 + wk(wc, k)) + 0.5 * np.sqrt((wk(wc, k) - w0)**2 + 4 * Omega_R**2 * (wk(wc, k) / wc) * np.cos(theta)**2)

def vk(wc, k):

    vel = np.zeros((len(k)), dtype = float)

    for i in range(1, len(k)):
        vel[i] = (omega_LP(wc, k[i + 1]) - omega_LP(wc, k[i - 1])) / (2 * (k[1] - k[0]))

    return vel[1:-1]

wc = 1.90 / conv                   # cavity frequency
# need to further add cavity losses
w0 = 1.96 / conv                   # exciton frequency
gamma = 0.0062 / conv             # characteristic frequency
ndof = 1000                       # number of bath DOFs

Nexc = 10001
Ncav = 283
L = 0.04 * mm2au                                                            # box lenght
dk = 2 * np.pi / L                                                          # k-space unit length
k_par = np.linspace(- (Nexc - 1) * dk / 2, (Nexc - 1) * dk / 2, Nexc)       # discretizing the whole band
k_plot = np.linspace(- (Ncav - 1) * dk / 2, (Ncav - 1) * dk / 2, Ncav)      # discretizing the polaritons
Omega_R = 0.12 / conv                                                       # value of \sqrt(N) gc
gammak = 0.001 / conv                                                       # value of 0_+

color11 = '#1868B2'
color22 = '#C5272D'
color33 = '#018A67'

# ==============================================================================================
#                           Fig S3a 
# ==============================================================================================

plt.subplot(1, 2, 1)

data5 = np.loadtxt("./Ehrenfest.txt", dtype = float)

k_plot2 = np.array([4.7568 * dk, 9.289885 * dk, 12.413037 * dk, 15.064053 * dk, 17.49634 * dk, 19.836106 * dk, 22.170975 * dk])
theta2 = np.arctan(k_plot2 / (wc / c2au))
sin_2_thetak2 = 0.5 + 0.5 * (wk(wc, k_plot2) - w0) / np.sqrt((wk(wc, k_plot2) - w0)**2 + 4 * Omega_R**2 * (wk(wc, k_plot2) / wc) * np.cos(theta2)**2)
v_bare = np.zeros((len(k_plot2)), dtype = float)
for i in range(len(k_plot2)):
    v_bare[i] = (omega_LP(wc, k_plot2[i] + 0.01 * dk) - omega_LP(wc, k_plot2[i] - 0.01 * dk)) / (0.02 * dk)
    
# plt.plot(sin_2_thetak2, v_bare / um_ps2au - data5[:, 1], 'o', markersize = 10, linewidth = 1, markerfacecolor = 'white', color = color11)
plt.plot(sin_2_thetak2, v_bare / um_ps2au - data5[:, 2], 'o', markersize = 10, linewidth = 1, markerfacecolor = 'white', color = color22, label = r'Ehrenfest')
# plt.plot(sin_2_thetak2, v_bare / um_ps2au - data5[:, 3], 'o', markersize = 10, linewidth = 1, markerfacecolor = 'white', color = color33)

theta = np.arctan(k_plot / (wc / c2au))
sin_2_thetak = 0.5 + 0.5 * (wk(wc, k_plot) - w0) / np.sqrt((wk(wc, k_plot) - w0)**2 + 4 * Omega_R**2 * (wk(wc, k_plot) / wc) * np.cos(theta)**2)

data1 = np.loadtxt("./dv_LP_lambda=12meV.txt", dtype = float)
plt.plot(sin_2_thetak[1:-1], data1[:, 3] / um_ps2au, '-', linewidth = lw, color = color22, label = r"Total")
plt.plot(sin_2_thetak[1:-1], data1[:, 1] / um_ps2au, '--', linewidth = lw / 2, color = color22, alpha = .7, label = r"Part I")
plt.plot(sin_2_thetak[1:-1], data1[:, 2] / um_ps2au, '-', linewidth = lw / 2, color = color22, alpha = .7, label = r"Part II")

# ==============================================================================================
# RHS y-axis
ax = plt.gca()
x_major_locator = MultipleLocator(0.1)
x_minor_locator = MultipleLocator(0.02)
y_major_locator = MultipleLocator(2)
y_minor_locator = MultipleLocator(1)
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

plt.tick_params("x", which = 'both', direction = 'in', labelsize = 30, pad = 16)
plt.tick_params("y", which = 'both', direction = 'in', labelsize = 30)
plt.xlim(0.4, 0.6)
plt.ylim(0, 7)

# RHS y-axis
ax2 = ax.twinx()
ax2.yaxis.set_major_locator(y_major_locator)
ax2.yaxis.set_minor_locator(y_minor_locator)
ax2.tick_params(which = 'major', length = 15)
ax2.tick_params(which = 'minor', length = 5)
ax2.axes.yaxis.set_ticklabels([])

plt.tick_params(which = 'both', direction = 'in')
plt.ylim(0, 7)

ax.set_xlabel(r'$|C_k|^2$', size = 32)
ax.set_ylabel(r'$\Delta v_{g,-}~(\mu \mathrm{m / ps})$', size = 32)
ax.legend(loc = 'upper left', frameon = False, prop = font_legend)
plt.legend(title = '(a)', bbox_to_anchor = (legend_x, legend_y), frameon = False, title_fontsize = legendsize)

# ==============================================================================================
#                           Fig S3b 
# ==============================================================================================

plt.subplot(1, 2, 2)

color1 = '#0095FF'
color2 = '#019092'
color3 = '#6FDCB5'

k1 = 9.289885 * dk          # E = 1.82 eV
k2 = 15.064053 * dk         # E = 1.84 eV
k3 = 19.836106 * dk         # E = 1.86 eV
window = 0.01 * dk          # window size for calculating numerical derivative
k_plot = np.array([k1, k2, k3]) 

v1 = (omega_LP(wc, k_plot[0] + 0.01 * dk) - omega_LP(wc, k_plot[0] - 0.01 * dk)) / (0.02 * dk)    # 1.82
v2 = (omega_LP(wc, k_plot[1] + 0.01 * dk) - omega_LP(wc, k_plot[1] - 0.01 * dk)) / (0.02 * dk)    # 1.84
v3 = (omega_LP(wc, k_plot[2] + 0.01 * dk) - omega_LP(wc, k_plot[2] - 0.01 * dk)) / (0.02 * dk)    # 1.86

data = np.loadtxt("./group_vel_LP.txt", dtype = float)

plt.plot(data[:, 0], (v2 - data[:, 1]) / um_ps2au, '-', linewidth = lw, color = color1, label = r"Total")
plt.plot(data[:, 0], (v2 - data[:, 2]) / um_ps2au, '-', linewidth = lw / 2, color = color2, label = r"Relaxation")
plt.plot(data[:, 0], (v2 - data[:, 3]) / um_ps2au, '-', linewidth = lw / 2, color = color3, label = r"Excitation")

# ==============================================================================================
# RHS y-axis
ax = plt.gca()
x_major_locator = MultipleLocator(100)
x_minor_locator = MultipleLocator(50)
y_major_locator = MultipleLocator(1)
y_minor_locator = MultipleLocator(0.2)
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

plt.tick_params("x", which = 'both', direction = 'in', labelsize = 30, pad = 16)
plt.tick_params("y", which = 'both', direction = 'in', labelsize = 30)
plt.xlim(0, 400)
plt.ylim(0, 2)

# RHS y-axis
ax2 = ax.twinx()
ax2.yaxis.set_major_locator(y_major_locator)
ax2.yaxis.set_minor_locator(y_minor_locator)
ax2.tick_params(which = 'major', length = 15)
ax2.tick_params(which = 'minor', length = 5)
ax2.axes.yaxis.set_ticklabels([])

plt.tick_params(which = 'both', direction = 'in')
plt.ylim(0, 2)

ax.set_xlabel(r'$T~(\mathrm{K})$', size = 32)
ax.set_ylabel(r'$\Delta v_{g,-}~(\mu \mathrm{m / ps})$', size = 32)
ax.legend(loc = 'upper left', frameon = False, prop = font_legend)
plt.legend(title = '(b)', bbox_to_anchor = (legend_x, legend_y), frameon = False, title_fontsize = legendsize)



plt.savefig("Fig_S3.pdf", bbox_inches='tight')