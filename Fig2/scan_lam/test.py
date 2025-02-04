import numpy as np
import matplotlib.pyplot as plt

conv = 27.211397                            # 1 a.u. = 27.211397 eV
au_to_K = 3.1577464e+05                     # 1 au = 3.1577464e+05 K
mm2au = 18897261.257078                     # 1 mm = 18897261.257078 a.u.
c2au = 137.0359895
fs2au = 41.341374575751                     # 1 fs = 41.341374575751 a.u.
um_ps2au = (mm2au / 1000) / (1000 * fs2au)  # um / ps in a.u.

wc = 1.90 / conv                   # cavity frequency
# need to further add cavity losses
w0 = 1.96 / conv                   # exciton frequency
gamma = 0.0062 / conv             # characteristic frequency
ndof = 10000                      # number of bath DOFs

Nexc = 10001
Ncav = 283
L = 0.04 * mm2au                                                            # box lenght
dk = 2 * np.pi / L                                                          # k-space unit length
k_par = np.linspace(- (Nexc - 1) * dk / 2, (Nexc - 1) * dk / 2, Nexc)       # discretizing the whole band
k1 = 9.289885 * dk          # E = 1.82 eV
k2 = 12.413037 * dk         # E = 1.83 eV
k3 = 15.064053 * dk         # E = 1.84 eV
window = 1.0 * dk          # window size for calculating numerical derivative
# k_plot = np.array([k1 - window, k1 + 0 * window, k2 - window, k2 + 0 * window, k3 - window, k3 + 0 * window]) 
k_plot = np.linspace(- (Ncav - 1) * dk / 2, (Ncav - 1) * dk / 2, Ncav)      # discretizing the polaritons
Omega_R = 0.12 / conv                                                       # value of \sqrt(N) gc
gammak = 0.001 / conv  

def Bose(x):
    beta = au_to_K / 300
    return 1 / (np.exp(beta * x) - 1)

def bathParam(λ, ωc, ndof):

    c = np.zeros(( ndof ))
    ω = np.zeros(( ndof ))
    for d in range(ndof):
        ω[d] =  ωc * np.tan(np.pi * (1 - (d + 1)/(ndof + 1)) / 2)
        c[d] =  np.sqrt(λ * ω[d] / (ndof + 1))

    return c, ω

def wk(wc, k):
    return np.sqrt(wc**2 + c2au**2 * k**2)

def omega_LP(wc, k):
    theta = np.arctan(k / (wc / c2au))
    return 0.5 * (w0 + wk(wc, k)) - 0.5 * np.sqrt((wk(wc, k) - w0)**2 + 4 * Omega_R**2 * (wk(wc, k) / wc) * np.cos(theta)**2)

def vk(wc, k):

    vel = np.zeros((len(k)), dtype = float)

    for i in range(1, len(k) - 1):
        vel[i] = (omega_LP(wc, k[i]) - omega_LP(wc, k[i - 1])) / ((k[1] - k[0]))

    return vel[1:-1]

def vk1(wc, k):

    vel = np.zeros((len(k) - 2), dtype = float)

    for i in range(1, len(k) - 1):
        vel[i - 1] = (omega_LP(wc, k[i + 1]) - omega_LP(wc, k[i - 1])) / (2 * (k[1] - k[0]))

    return vel

def vk2(wc, k):

    vel = np.zeros((len(k)), dtype = float)

    for i in range(1, len(k) - 1):
        vel[i] = (omega_LP(wc, k[i] + 0.01 * dk) - omega_LP(wc, k[i] - 0.01 * dk)) / (0.02 * dk)

    return vel[1:-1]

plt.plot(omega_LP(wc, k_plot[1:-1]) * conv, vk(wc, k_plot) / um_ps2au, label = 'v[k] - v[k-1] / dk')
plt.plot(omega_LP(wc, k_plot[1:-1]) * conv, vk1(wc, k_plot) / um_ps2au, label = 'v[k+1] - v[k-1] / 2dk')
plt.plot(omega_LP(wc, k_plot[1:-1]) * conv, vk2(wc, k_plot) / um_ps2au, label = 'benchmark')

plt.xlim(1.80, 1.88)
plt.ylim(0, 45)
plt.legend()
plt.show()
