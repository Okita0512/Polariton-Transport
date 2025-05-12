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
fig = plt.figure(figsize=(2.5 * unitlen, 0.9 * unitlen), dpi = 128)
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

    for i in range(1, len(k) - 1):
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

# ==============================================================================================
#                           Fig 2a 
# ==============================================================================================

plt.subplot(1, 2, 1)

color00 = 'purple'
color11 = '#1868B2'
color22 = '#C5272D'
color33 = '#018A67'

datam = np.loadtxt("./3-1meV/band_struct.txt", dtype = float)
data0 = np.loadtxt("./6-2meV/band_struct.txt", dtype = float)
data1 = np.loadtxt("./12-4meV/band_struct.txt", dtype = float)
data2 = np.loadtxt("./18-6meV/band_struct.txt", dtype = float)
# data3 = np.loadtxt("./36meV/band_struct_LP.txt", dtype = float)
# data4 = np.loadtxt("./48meV/band_struct_LP.txt", dtype = float)

plt.plot(k_par / dk, omega_LP(wc, k_par) * conv, '-', linewidth = lw, color = 'k', label = r"No Bath")
plt.plot(datam[:, 0] / dk, datam[:, 1] * conv, '-', linewidth = lw, color = color00, label = r"$\omega_f$ = 3.1 meV")
plt.plot(data0[:, 0] / dk, data0[:, 1] * conv, '-', linewidth = lw, color = color11, label = r"$\omega_f$ = 6.2 meV")
plt.plot(data1[:, 0] / dk, data1[:, 1] * conv, '-', linewidth = lw, color = color22, label = r"$\omega_f$ = 12.4 meV")
plt.plot(data2[:, 0] / dk, data2[:, 1] * conv, '-', linewidth = lw, color = color33, label = r"$\omega_f$ = 18.6 meV")

plt.plot(k_par / dk, omega_UP(wc, k_par) * conv, '-', linewidth = lw, color = 'k')
plt.plot(datam[:, 0] / dk, datam[:, 2] * conv, '-', linewidth = lw, color = color00)
plt.plot(data0[:, 0] / dk, data0[:, 2] * conv, '-', linewidth = lw, color = color11)
plt.plot(data1[:, 0] / dk, data1[:, 2] * conv, '-', linewidth = lw, color = color22)
plt.plot(data2[:, 0] / dk, data2[:, 2] * conv, '-', linewidth = lw, color = color33)

plt.plot(k_par / dk, wk(wc, k_par) * conv, '--', linewidth = 2.0, color = 'k', alpha = .4)
plt.hlines([1.96], -75, 75, linestyles= ['--'], linewidth = 2.0, color = 'k', alpha = .4)

# ==============================================================================================

# RHS y-axis
ax = plt.gca()
x_major_locator = MultipleLocator(25)
x_minor_locator = MultipleLocator(5)
y_major_locator = MultipleLocator(0.1)
y_minor_locator = MultipleLocator(0.02)
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
plt.xlim(-75, 75)
plt.ylim(1.76, 2.2)

# RHS y-axis
ax2 = ax.twinx()
ax2.yaxis.set_major_locator(y_major_locator)
ax2.yaxis.set_minor_locator(y_minor_locator)
ax2.tick_params(which = 'major', length = 15)
ax2.tick_params(which = 'minor', length = 5)
ax2.axes.yaxis.set_ticklabels([])

plt.tick_params(which = 'both', direction = 'in')
plt.ylim(1.76, 2.2)

ax.set_xlabel(r'$k~(2\pi / L)$', size = 32)
ax.set_ylabel(r'Energy $(\mathrm{eV})$', size = 32)
# ax.legend(loc = 'upper center', frameon = False, prop = font_legends)

plt.legend(title = '(a)', bbox_to_anchor = (legend_x, legend_y), frameon = False, title_fontsize = legendsize)

# ==============================================================================================
#                           Fig 2b 
# ==============================================================================================

plt.subplot(1, 2, 2)

datam = np.loadtxt("./3-1meV/group_vel_LP.txt", dtype = float)
data0 = np.loadtxt("./6-2meV/group_vel_LP.txt", dtype = float)
data1 = np.loadtxt("./12-4meV/group_vel_LP.txt", dtype = float)
data2 = np.loadtxt("./18-6meV/group_vel_LP.txt", dtype = float)

data5 = np.loadtxt("./Ehrenfest.txt", dtype = float)

plt.plot(omega_LP(wc, k_par)[1:-1] * conv, vk(wc, k_par) / um_ps2au, '-', linewidth = lw, color = 'k', label = r"No Bath")

plt.plot(datam[:, 0] * conv, datam[:, 1] / um_ps2au, '-', linewidth = lw, color = color00, label = r"$\omega_f$ = 3.1 meV")
plt.plot(data0[:, 0] * conv, data0[:, 1] / um_ps2au, '-', linewidth = lw, color = color11, label = r"$\omega_f$ = 6.2 meV")
plt.plot(data1[:, 0] * conv, data1[:, 1] / um_ps2au, '-', linewidth = lw, color = color22, label = r"$\omega_f$ = 12.4 meV")
plt.plot(data2[:, 0] * conv, data2[:, 1] / um_ps2au, '-', linewidth = lw, color = color33, label = r"$\omega_f$ = 18.6 meV")

plt.plot(data5[:, 0], data5[:, 1], 'o', markersize = 10, linewidth = 1, markerfacecolor = 'white', color = color00)
plt.plot(data5[:, 0], data5[:, 2], 'o', markersize = 10, linewidth = 1, markerfacecolor = 'white', color = color11)
plt.plot(data5[:, 0], data5[:, 3], 'o', markersize = 10, linewidth = 1, markerfacecolor = 'white', color = color22)
plt.plot(data5[:, 0], data5[:, 4], 'o', markersize = 10, linewidth = 1, markerfacecolor = 'white', color = color33)

# ==============================================================================================
# RHS y-axis
ax = plt.gca()
x_major_locator = MultipleLocator(0.02)
x_minor_locator = MultipleLocator(0.01)
y_major_locator = MultipleLocator(10)
y_minor_locator = MultipleLocator(2)
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
plt.xlim(1.80, 1.88)
plt.ylim(0, 44)

# RHS y-axis
ax2 = ax.twinx()
ax2.yaxis.set_major_locator(y_major_locator)
ax2.yaxis.set_minor_locator(y_minor_locator)
ax2.tick_params(which = 'major', length = 15)
ax2.tick_params(which = 'minor', length = 5)
ax2.axes.yaxis.set_ticklabels([])

plt.tick_params(which = 'both', direction = 'in')
plt.ylim(0, 44)

ax.set_xlabel(r'Energy $(\mathrm{eV})$', size = 32)
ax.set_ylabel(r'$v_{g,-}~(\mu \mathrm{m / ps})$', size = 32)
ax.legend(loc = 'lower center', frameon = False, prop = font_legend)
plt.legend(title = '(b)', bbox_to_anchor = (legend_x, legend_y), frameon = False, title_fontsize = legendsize)





plt.savefig("Fig_S8.pdf", bbox_inches='tight')