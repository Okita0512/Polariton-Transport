import numpy as np

conv = 27.211397                            # 1 a.u. = 27.211397 eV
au_to_K = 3.1577464e+05                     # 1 au = 3.1577464e+05 K
mm2au = 18897261.257078                     # 1 mm = 18897261.257078 a.u.
c2au = 137.0359895

def Bose(T, x):
    beta = au_to_K / T
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
k1 = 9.289885 * dk          # E = 1.82 eV
k2 = 15.064053 * dk         # E = 1.84 eV
k3 = 19.836106 * dk         # E = 1.86 eV
window = 0.01 * dk          # window size for calculating numerical derivative
k_plot = np.array([k1 - window, k1 + window, k2 - window, k2 + window, k3 - window, k3 + window]) 
# k_plot = np.linspace(- (Ncav - 1) * dk / 2, (Ncav - 1) * dk / 2, Ncav)      # discretizing the polaritons
Omega_R = 0.12 / conv                                                       # value of \sqrt(N) gc
gammak = 0.001 / conv                                                       # value of 0_+

# set k_plot to satisfy E = 1.82, 1.84, 1.86 eV, and change lambda continuously from 0 to 24 meV
# print(omega_LP(wc, 9.289885 * dk) * conv)
# print(omega_LP(wc, 15.064053 * dk) * conv)
# print(omega_LP(wc, 19.836106 * dk) * conv)

lam = 0.006 / conv
w0 = w0 # + lam                      # further adding the reorganization energy
cb, wb = bathParam(lam, gamma, ndof)
temp = np.linspace(0, 400, 200)

tilde_wk1 = np.zeros((len(temp), len(k_plot)), dtype = float)
vel = np.zeros((len(temp), 3), dtype = float)

for Ti in range(len(temp)):
    
    tmp1 = np.zeros((len(k_plot)), dtype = float)   # LP renormalization
#    tmp2 = np.zeros((len(k_plot)), dtype = float)   # UP renormalization

    count = 0
    for kj in k_plot:

            # major contribution
            # DS contribution
            tmp1[count] += np.real(np.sum(cb**2 * (1 + Bose(temp[Ti], wb)) / ((omega_LP(wc, kj) - w0 - wb + 1.0j * gammak))
                                  + cb**2 * Bose(temp[Ti], wb) / ((omega_LP(wc, kj) - w0 + wb + 1.0j * gammak))))
            
#                tmp2[count] += np.real(np.sum(cb**2 * (1 + Bose(wb)) / (Nexc * (omega_UP(wc, kj) - w0 - wb + 1.0j * gammak))
#                                  + cb**2 * Bose(wb) / (Nexc * (omega_UP(wc, kj) - w0 + wb + 1.0j * gammak))))
        

            count += 1

    theta = np.arctan(k_plot / (wc / c2au))
    sin_2_thetak = 0.5 + 0.5 * (wk(wc, k_plot) - w0) / np.sqrt((wk(wc, k_plot) - w0)**2 + 4 * Omega_R**2 * (wk(wc, k_plot) / wc) * np.cos(theta)**2)
    # cos_2_thetak = 0.5 - 0.5 * (wk(wc, k_plot) - w0) / np.sqrt((wk(wc, k_plot) - w0)**2 + 4 * Omega_R**2 * (wk(wc, k_plot) / wc) * np.cos(theta)**2)

    tilde_wk1[Ti, :] = omega_LP(wc, k_plot) + 2 * tmp1 * sin_2_thetak

    vel[Ti, 0] = (tilde_wk1[Ti, 1] - tilde_wk1[Ti, 0]) / (k_plot[1] - k_plot[0])
    vel[Ti, 1] = (tilde_wk1[Ti, 3] - tilde_wk1[Ti, 2]) / (k_plot[3] - k_plot[2])
    vel[Ti, 2] = (tilde_wk1[Ti, 5] - tilde_wk1[Ti, 4]) / (k_plot[5] - k_plot[4])

group_vel_LP = np.zeros((len(temp), 4), dtype = float)
group_vel_LP[:, 0] = temp
group_vel_LP[:, 1:] = vel

np.savetxt("group_vel_LP.txt", group_vel_LP)
