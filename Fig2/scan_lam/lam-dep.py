import numpy as np

conv = 27.211397                            # 1 a.u. = 27.211397 eV
au_to_K = 3.1577464e+05                     # 1 au = 3.1577464e+05 K
mm2au = 18897261.257078                     # 1 mm = 18897261.257078 a.u.
c2au = 137.0359895

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
ndof = 10000                      # number of bath DOFs

Nexc = 10001
Ncav = 283
L = 0.04 * mm2au                                                            # box lenght
dk = 2 * np.pi / L                                                          # k-space unit length
k_par = np.linspace(- (Nexc - 1) * dk / 2, (Nexc - 1) * dk / 2, Nexc)       # discretizing the whole band
k1 = 9.289885 * dk          # E = 1.82 eV
k2 = 12.413037 * dk         # E = 1.83 eV
k3 = 15.064053 * dk         # E = 1.84 eV
window = 0.01 * dk          # window size for calculating numerical derivative
k_plot = np.array([k1 - window, k1 + window, k2 - window, k2 + window, k3 - window, k3 + window]) 
# k_plot = np.linspace(- (Ncav - 1) * dk / 2, (Ncav - 1) * dk / 2, Ncav)      # discretizing the polaritons
Omega_R = 0.12 / conv                                                       # value of \sqrt(N) gc
gammak = 0.001 / conv                                                       # value of 0_+

# set k_plot to satisfy E = 1.82, 1.83, 1.84 eV, and change lambda continuously from 0 to 24 meV
# print(omega_LP(wc, 9.289885 * dk) * conv)
print(omega_LP(wc, 12.413037 * dk) * conv)
# print(omega_LP(wc, 15.064053 * dk) * conv)

lam = np.linspace(0, 0.024 / conv, 100)

tilde_wk1 = np.zeros((len(lam), len(k_plot)), dtype = float)
vel = np.zeros((len(lam), 3), dtype = float)

for i in range(len(lam)):
    w0 = w0 # + lam[i]                      # further adding the reorganization energy
    cb, wb = bathParam(lam[i], gamma, ndof)

    tmp1 = np.zeros((len(k_plot)), dtype = float)   # LP renormalization
#    tmp2 = np.zeros((len(k_plot)), dtype = float)   # UP renormalization

    count = 0
    for kj in k_plot:

        # major contribution
        tmp1[count] += np.real(np.sum(cb**2 * (1 + Bose(wb)) / ((omega_LP(wc, kj) - w0 - wb + 1.0j * gammak))
                            + cb**2 * Bose(wb) / ((omega_LP(wc, kj) - w0 + wb + 1.0j * gammak))))
        
        count += 1

    theta = np.arctan(k_plot / (wc / c2au))
    sin_2_thetak = 0.5 + 0.5 * (wk(wc, k_plot) - w0) / np.sqrt((wk(wc, k_plot) - w0)**2 + 4 * Omega_R**2 * (wk(wc, k_plot) / wc) * np.cos(theta)**2)
    # cos_2_thetak = 0.5 - 0.5 * (wk(wc, k_plot) - w0) / np.sqrt((wk(wc, k_plot) - w0)**2 + 4 * Omega_R**2 * (wk(wc, k_plot) / wc) * np.cos(theta)**2)

    tilde_wk1[i, :] = omega_LP(wc, k_plot) + 2 * tmp1 * sin_2_thetak

    vel[i, 0] = (tilde_wk1[i, 1] - tilde_wk1[i, 0]) / (k_plot[1] - k_plot[0])
    vel[i, 1] = (tilde_wk1[i, 3] - tilde_wk1[i, 2]) / (k_plot[3] - k_plot[2])
    vel[i, 2] = (tilde_wk1[i, 5] - tilde_wk1[i, 4]) / (k_plot[5] - k_plot[4])

group_vel_LP = np.zeros((len(lam), 4), dtype = float)
group_vel_LP[:, 0] = lam
group_vel_LP[:, 1] = vel[:, 0]
group_vel_LP[:, 2] = vel[:, 1]
group_vel_LP[:, 3] = vel[:, 2]

np.savetxt("group_vel_LP.txt", group_vel_LP)
