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

    for i in range(1, len(k)):
        vel[i] = (omega_LP(wc, k[i]) - omega_LP(wc, k[i - 1])) / (k[1] - k[0])

    return vel[:-1]

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
k_plot = np.linspace(- (Ncav - 1) * dk / 2, (Ncav - 1) * dk / 2, Ncav)      # discretizing the polaritons
Omega_R = 0.12 / conv                                                       # value of \sqrt(N) gc
gammak = 0.001 / conv                                                       # value of 0_+

# case 1
lam = 0.012 / conv                 # reorganization energy
w0 = w0 # + lam                      # further adding the reorganization energy
cb, wb = bathParam(lam, gamma, ndof)

tmp1 = np.zeros((len(k_plot)), dtype = float)   # LP renormalization

count = 0
for kj in k_plot:
        
    # DS contribution
    tmp1[count] += np.real(np.sum(cb**2 * (1 + Bose(wb)) / (Nexc * (omega_LP(wc, kj) - w0 - wb + 1.0j * gammak))
                    + cb**2 * Bose(wb) / (Nexc * (omega_LP(wc, kj) - w0 + wb + 1.0j * gammak)))) * (Nexc - Ncav)
    
    count += 1

theta = np.arctan(k_plot / (wc / c2au))
sin_2_thetak = 0.5 + 0.5 * (wk(wc, k_plot) - w0) / np.sqrt((wk(wc, k_plot) - w0)**2 + 4 * Omega_R**2 * (wk(wc, k_plot) / wc) * np.cos(theta)**2)
cos_2_thetak = 0.5 - 0.5 * (wk(wc, k_plot) - w0) / np.sqrt((wk(wc, k_plot) - w0)**2 + 4 * Omega_R**2 * (wk(wc, k_plot) / wc) * np.cos(theta)**2)

tilde_wk1 = omega_LP(wc, k_plot) + 2 * tmp1 * sin_2_thetak

dCk = np.zeros(len(k_plot), dtype = float)
for i in range(1, len(k_plot) - 1):
    dCk[i] = (sin_2_thetak[i+1] - sin_2_thetak[i-1]) / (2 * (k_plot[1] - k_plot[0]))

dHk = np.zeros(len(k_plot), dtype = float)
for i in range(1, len(k_plot) - 1):
    dHk[i] = (tmp1[i+1] - tmp1[i-1]) / (2 * (k_plot[1] - k_plot[0]))

dv_LP_1 = 2 * tmp1 * dCk
dv_LP_2 = 2 * dHk * sin_2_thetak
dv_LP_3 = dv_LP_1 + dv_LP_2

group_vel_LP = np.zeros((Ncav - 2, 4), dtype = float)
group_vel_LP[:, 0] = omega_LP(wc, k_plot)[1:-1]
group_vel_LP[:, 1] = dv_LP_1[1:-1]
group_vel_LP[:, 2] = dv_LP_2[1:-1]
group_vel_LP[:, 3] = dv_LP_3[1:-1]

np.savetxt("dv_LP_lambda=12meV.txt", group_vel_LP)
