import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator, tick_params
fig, ax = plt.subplots()
plt.rcParams['font.family'] = 'DeJavu Serif'
plt.rcParams["font.family"] = "Helvetica"

# ==============================================================================================
#                                       Global Parameters     
# ==============================================================================================
lw = 3.0
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

    for i in range(1, len(k)):
        vel[i] = (omega_LP(wc, k[i]) - omega_LP(wc, k[i - 1])) / (k[1] - k[0])

    return vel[:-1]

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

v1 = (omega_LP(wc, k_plot[0]) - omega_LP(wc, k_plot[0] - dk)) / (dk)    # 1.82
v2 = (omega_LP(wc, k_plot[1]) - omega_LP(wc, k_plot[1] - dk)) / (dk)    # 1.84
v3 = (omega_LP(wc, k_plot[2]) - omega_LP(wc, k_plot[2] - dk)) / (dk)    # 1.86

v11 = (omega_LP(wc, k_plot[0]) - omega_LP(wc, k_plot[0] - 0.29 * dk)) / (0.29 * dk)
v22 = (omega_LP(wc, k_plot[1]) - omega_LP(wc, k_plot[1] - 0.06 * dk)) / (0.06 * dk)
v33 = (omega_LP(wc, k_plot[2]) - omega_LP(wc, k_plot[2] - 0.83 * dk)) / (0.83 * dk)

print("error v1:", (v1 - v11) / um_ps2au)
print("error v2:", (v2 - v22) / um_ps2au)
print("error v3:", (v3 - v33) / um_ps2au)