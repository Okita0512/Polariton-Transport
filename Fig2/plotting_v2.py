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
fig = plt.figure(figsize=(2.5 * unitlen, 2.0 * unitlen), dpi = 128)
plt.subplots_adjust(hspace = 0.25, wspace = 0.32)

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
dk0 = dk * (40 / (2 * np.pi))                                                # convert to micron^-1
k_par = np.linspace(- (Nexc - 1) * dk / 2, (Nexc - 1) * dk / 2, Nexc)       # discretizing the whole band
k_plot = np.linspace(- (Ncav - 1) * dk / 2, (Ncav - 1) * dk / 2, Ncav)      # discretizing the polaritons
Omega_R = 0.12 / conv                                                       # value of \sqrt(N) gc
gammak = 0.001 / conv                                                       # value of 0_+

# ==============================================================================================
#                           Fig 2a 
# ==============================================================================================

plt.subplot(2, 2, 1)

color11 = '#1868B2'
color22 = '#C5272D'
color33 = '#018A67'

data0 = np.loadtxt("./6meV/band_struct_LP.txt", dtype = float)
data1 = np.loadtxt("./12meV/band_struct_LP.txt", dtype = float)
data2 = np.loadtxt("./24meV/band_struct_LP.txt", dtype = float)
# data3 = np.loadtxt("./36meV/band_struct_LP.txt", dtype = float)
# data4 = np.loadtxt("./48meV/band_struct_LP.txt", dtype = float)

plt.plot(k_par / dk0, omega_LP(wc, k_par) * conv, '-', linewidth = lw, color = 'k', label = r"No Bath")
plt.plot(data0[:, 0] / dk0, data0[:, 1] * conv, '-', linewidth = lw, color = color11, label = r"$\lambda$ = 6 meV")
plt.plot(data1[:, 0] / dk0, data1[:, 1] * conv, '-', linewidth = lw, color = color22, label = r"$\lambda$ = 12 meV")
plt.plot(data2[:, 0] / dk0, data2[:, 1] * conv, '-', linewidth = lw, color = color33, label = r"$\lambda$ = 24 meV")
# plt.plot(data3[:, 0] / dk, data3[:, 1] * conv, '-', linewidth = lw, color = 'green', label = r"$\lambda$ = 36 meV")
# plt.plot(data4[:, 0] / dk, data4[:, 1] * conv, '-', linewidth = lw, color = 'purple', label = r"$\lambda$ = 48 meV")

plt.plot(k_par / dk0, omega_UP(wc, k_par) * conv, '-', linewidth = lw, color = 'k') #, label = r"No Bath")
plt.plot(data0[:, 0] / dk0, data0[:, 2] * conv, '-', linewidth = lw, color = color11) #, label = r"$\lambda$ = 6 meV")
plt.plot(data1[:, 0] / dk0, data1[:, 2] * conv, '-', linewidth = lw, color = color22)# , label = r"$\lambda$ = 12 meV")
plt.plot(data2[:, 0] / dk0, data2[:, 2] * conv, '-', linewidth = lw, color = color33)# , label = r"$\lambda$ = 24 meV")
# plt.plot(data3[:, 0] / dk, data3[:, 2] * conv, '-', linewidth = lw, color = 'green', label = r"$\lambda$ = 36 meV")
# plt.plot(data4[:, 0] / dk, data4[:, 2] * conv, '-', linewidth = lw, color = 'purple', label = r"$\lambda$ = 48 meV")

plt.plot(k_par / dk0, wk(wc, k_par) * conv, '--', linewidth = 2.0, color = 'k', alpha = .4)
plt.hlines([1.96], -12, 12, linestyles= ['--'], linewidth = 2.0, color = 'k', alpha = .4)

# ==============================================================================================

# RHS y-axis
ax = plt.gca()
x_major_locator = MultipleLocator(5)
x_minor_locator = MultipleLocator(1)
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
plt.xlim(-12, 12)
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

ax.set_xlabel(r'$k_\parallel$ ($\mu$m$^{-1}$)', size = 32)
ax.set_ylabel(r'Energy $(\mathrm{eV})$', size = 32)
# ax.legend(loc = 'upper center', frameon = False, prop = font_legends)

plt.legend(title = '(a)', bbox_to_anchor = (legend_x, legend_y), frameon = False, title_fontsize = legendsize)

# ==============================================================================================
#                           Fig 2b 
# ==============================================================================================

plt.subplot(2, 2, 2)

data0 = np.loadtxt("./6meV/group_vel_LP.txt", dtype = float)
data1 = np.loadtxt("./12meV/group_vel_LP.txt", dtype = float)
data2 = np.loadtxt("./24meV/group_vel_LP.txt", dtype = float)
# data3 = np.loadtxt("./36meV/group_vel_LP.txt", dtype = float)
# data4 = np.loadtxt("./48meV/group_vel_LP.txt", dtype = float)

data5 = np.loadtxt("./Ehrenfest.txt", dtype = float)

plt.plot(omega_LP(wc, k_par)[1:-1] * conv, vk(wc, k_par) / um_ps2au, '-', linewidth = lw, color = 'k', label = r"No Bath")

plt.plot(data0[:, 0] * conv, data0[:, 1] / um_ps2au, '-', linewidth = lw, color = color11, label = r"$\lambda$ = 6 meV")
plt.plot(data1[:, 0] * conv, data1[:, 1] / um_ps2au, '-', linewidth = lw, color = color22, label = r"$\lambda$ = 12 meV")
plt.plot(data2[:, 0] * conv, data2[:, 1] / um_ps2au, '-', linewidth = lw, color = color33, label = r"$\lambda$ = 24 meV")
# plt.plot(data4[:, 0] * conv, data4[:, 1] / um_ps2au, '-', linewidth = lw, color = 'purple', label = r"$\lambda$ = 48 meV")

plt.plot(data5[:, 0], data5[:, 1], 'o', markersize = 10, linewidth = 1, markerfacecolor = 'white', color = color11)
plt.plot(data5[:, 0], data5[:, 2], 'o', markersize = 10, linewidth = 1, markerfacecolor = 'white', color = color22)
plt.plot(data5[:, 0], data5[:, 3], 'o', markersize = 10, linewidth = 1, markerfacecolor = 'white', color = color33)
# plt.plot(data5[:, 0], data5[:, 4], 'o', markersize = 10, linewidth = 1, markerfacecolor = 'white', color = 'purple')

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

ax.set_xlabel(r'LP Energy $(\mathrm{eV})$', size = 32)
ax.set_ylabel(r'$\tilde{v}_{g,-}~(\mu \mathrm{m / ps})$', size = 32)
ax.legend(loc = 'lower center', frameon = False, prop = font_legend)
plt.legend(title = '(b)', bbox_to_anchor = (legend_x, legend_y), frameon = False, title_fontsize = legendsize)

# ==============================================================================================
#                           Fig 2b 
# ==============================================================================================

plt.subplot(2, 2, 3)

data = np.loadtxt("./scan_lam/group_vel_LP.txt", dtype = float)
k_plot = np.array([9.289885 * dk, 15.064053 * dk, 19.836106 * dk])

v1 = (omega_LP(wc, k_plot[0] + 0.01 * dk) - omega_LP(wc, k_plot[0] - 0.01 * dk)) / (0.02 * dk)    # 1.82
v2 = (omega_LP(wc, k_plot[1] + 0.01 * dk) - omega_LP(wc, k_plot[1] - 0.01 * dk)) / (0.02 * dk)    # 1.84
v3 = (omega_LP(wc, k_plot[2] + 0.01 * dk) - omega_LP(wc, k_plot[2] - 0.01 * dk)) / (0.02 * dk)    # 1.86

color1 = '#0095FF'
color2 = '#019092'
color3 = '#6FDCB5'

plt.plot(data[:, 0] * conv * 1000, data[:, 1] / um_ps2au, '-', linewidth = lw, color = color1, label = r"$\epsilon_{-k}$ = 1.82 eV")
plt.plot(data[:, 0] * conv * 1000, data[:, 2] / um_ps2au, '-', linewidth = lw, color = color2, label = r"$\epsilon_{-k}$ = 1.83 eV")
plt.plot(data[:, 0] * conv * 1000, data[:, 3] / um_ps2au, '-', linewidth = lw, color = color3, label = r"$\epsilon_{-k}$ = 1.84 eV")

data5 = np.loadtxt("./Ehren_lam_scan.txt", dtype = float)
plt.plot(data5[:, 0], data5[:, 1], 'o', markersize = 10, linewidth = 1, markerfacecolor = 'white', color = color1)
plt.plot(data5[:, 0], data5[:, 2], 'o', markersize = 10, linewidth = 1, markerfacecolor = 'white', color = color2)
plt.plot(data5[:, 0], data5[:, 3], 'o', markersize = 10, linewidth = 1, markerfacecolor = 'white', color = color3)

# ==============================================================================================
# RHS y-axis
ax = plt.gca()
x_major_locator = MultipleLocator(5)
x_minor_locator = MultipleLocator(1)
y_major_locator = MultipleLocator(5)
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
plt.xlim(0, 20)
plt.ylim(23, 45)

# RHS y-axis
ax2 = ax.twinx()
ax2.yaxis.set_major_locator(y_major_locator)
ax2.yaxis.set_minor_locator(y_minor_locator)
ax2.tick_params(which = 'major', length = 15)
ax2.tick_params(which = 'minor', length = 5)
ax2.axes.yaxis.set_ticklabels([])

plt.tick_params(which = 'both', direction = 'in')
plt.ylim(23, 45)

ax.set_xlabel(r'$\lambda~(\mathrm{meV})$', size = 32)
ax.set_ylabel(r'$\tilde{v}_{g,-}~(\mu \mathrm{m / ps})$', size = 32)
ax.legend(loc = 'upper right', frameon = False, prop = font_legend)

plt.legend(title = '(c)', bbox_to_anchor = (legend_x, legend_y), frameon = False, title_fontsize = legendsize)

# ==============================================================================================
#                           Fig 2b 
# ==============================================================================================

plt.subplot(2, 2, 4)

data5 = np.loadtxt("./Ehrenfest_temp.txt", dtype = float)
plt.plot(data5[:, 0], data5[:, 1], 'o', markersize = 10, linewidth = 1, markerfacecolor = 'white', color = color3, label = "Ehrenfest")

data = np.loadtxt("./scan_temp_0_400/group_vel_LP.txt", dtype = float)

G = 3.0
plt.plot(data[:, 0], data[:, 3] / um_ps2au, '-', linewidth = lw, color = color3, label = r"Current Theory")
plt.plot(data[1:, 0], (v3 / um_ps2au) / (1 + G * np.exp(- (au_to_K / data[1:, 0]) * (w0 - 1.86 / conv))), '--', linewidth = lw, color = "red", label = r"TAST", alpha = .5)

# ==============================================================================================
# RHS y-axis
ax = plt.gca()
x_major_locator = MultipleLocator(100)
x_minor_locator = MultipleLocator(50)
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
plt.xlim(0, 400)
plt.ylim(38, 42)

# RHS y-axis
ax2 = ax.twinx()
ax2.yaxis.set_major_locator(y_major_locator)
ax2.yaxis.set_minor_locator(y_minor_locator)
ax2.tick_params(which = 'major', length = 15)
ax2.tick_params(which = 'minor', length = 5)
ax2.axes.yaxis.set_ticklabels([])

plt.tick_params(which = 'both', direction = 'in')
plt.ylim(38, 42)

ax.set_xlabel(r'$T~(\mathrm{K})$', size = 32)
ax.set_ylabel(r'$\tilde{v}_{g,-}~(\mu \mathrm{m / ps})$', size = 32)
ax.legend(loc = 'lower left', frameon = False, prop = font_legend)
plt.legend(title = '(d)', bbox_to_anchor = (legend_x, legend_y), frameon = False, title_fontsize = legendsize)



plt.savefig("Fig_2.pdf", bbox_inches='tight')