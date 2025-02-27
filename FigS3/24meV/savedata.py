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
lam = 0.024 / conv                 # reorganization energy
w0 = w0 # + lam                      # further adding the reorganization energy
cb, wb = bathParam(lam, gamma, ndof)

tmp1 = np.zeros((len(k_plot)), dtype = float)   # LP renormalization
tmp2 = np.zeros((len(k_plot)), dtype = float)   # UP renormalization

count = 0
for kj in k_plot:
    for ki in k_par:

        # major contribution
        if (ki >= (Ncav - 1) * dk / 2) or (ki <= - (Ncav - 1) * dk / 2):
            # DS contribution
            tmp1[count] += np.real(np.sum(cb**2 * (1 + Bose(wb)) / (Nexc * (omega_LP(wc, kj) - w0 - wb + 1.0j * gammak))
                                  + cb**2 * Bose(wb) / (Nexc * (omega_LP(wc, kj) - w0 + wb + 1.0j * gammak))))
            
            tmp2[count] += np.real(np.sum(cb**2 * (1 + Bose(wb)) / (Nexc * (omega_UP(wc, kj) - w0 - wb + 1.0j * gammak))
                                  + cb**2 * Bose(wb) / (Nexc * (omega_UP(wc, kj) - w0 + wb + 1.0j * gammak))))
        
        # minor contribution
        if (- (Ncav - 1) * dk / 2 < ki < (Ncav - 1) * dk / 2):
            
            theta = np.arctan(ki / (wc / c2au))
            sin_2_theta = 0.5 + 0.5 * (wk(wc, ki) - w0) / np.sqrt((wk(wc, ki) - w0)**2 + 4 * Omega_R**2 * (wk(wc, ki) / wc) * np.cos(theta)**2)
            cos_2_theta = 0.5 - 0.5 * (wk(wc, ki) - w0) / np.sqrt((wk(wc, ki) - w0)**2 + 4 * Omega_R**2 * (wk(wc, ki) / wc) * np.cos(theta)**2)
            
            # LP contributions
            tmp1[count] += np.real(np.sum(sin_2_theta * wb * cb**2 * (1 + Bose(wb)) / (Nexc * (omega_LP(wc, kj) - omega_LP(wc, ki) - wb + 1.0j * gammak))
                                  + sin_2_theta * wb * cb**2 * Bose(wb) / (Nexc * (omega_LP(wc, kj) - omega_LP(wc, ki) + wb + 1.0j * gammak))))
            
            tmp2[count] += np.real(np.sum(sin_2_theta * wb * cb**2 * (1 + Bose(wb)) / (Nexc * (omega_UP(wc, kj) - omega_LP(wc, ki) - wb + 1.0j * gammak))
                                  + sin_2_theta * wb * cb**2 * Bose(wb) / (Nexc * (omega_UP(wc, kj) - omega_LP(wc, ki) + wb + 1.0j * gammak))))
            
            # UP contribution
            tmp1[count] += np.real(np.sum(cos_2_theta * wb * cb**2 * (1 + Bose(wb)) / (Nexc * (omega_LP(wc, kj) - omega_UP(wc, ki) - wb + 1.0j * gammak))
                                  + sin_2_theta * wb * cb**2 * Bose(wb) / (Nexc * (omega_LP(wc, kj) - omega_UP(wc, ki) + wb + 1.0j * gammak))))
            
            tmp2[count] += np.real(np.sum(cos_2_theta * wb * cb**2 * (1 + Bose(wb)) / (Nexc * (omega_UP(wc, kj) - omega_UP(wc, ki) - wb + 1.0j * gammak))
                                  + sin_2_theta * wb * cb**2 * Bose(wb) / (Nexc * (omega_UP(wc, kj) - omega_UP(wc, ki) + wb + 1.0j * gammak))))
    
    count += 1

theta = np.arctan(k_plot / (wc / c2au))
sin_2_thetak = 0.5 + 0.5 * (wk(wc, k_plot) - w0) / np.sqrt((wk(wc, k_plot) - w0)**2 + 4 * Omega_R**2 * (wk(wc, k_plot) / wc) * np.cos(theta)**2)
cos_2_thetak = 0.5 - 0.5 * (wk(wc, k_plot) - w0) / np.sqrt((wk(wc, k_plot) - w0)**2 + 4 * Omega_R**2 * (wk(wc, k_plot) / wc) * np.cos(theta)**2)

tilde_wk1 = omega_LP(wc, k_plot) + 2 * tmp1 * sin_2_thetak
tilde_wk2 = omega_UP(wc, k_plot) + 2 * tmp2 * cos_2_thetak

band_struct = np.zeros((Ncav, 3), dtype = float)
band_struct[:, 0] = k_plot
band_struct[:, 1] = tilde_wk1
band_struct[:, 2] = tilde_wk2

np.savetxt("band_struct_LP.txt", band_struct)

def vk_tilde(tilde_wk, k):

    vel = np.zeros((len(k)), dtype = float)

    for i in range(1, len(k)):
        vel[i] = (tilde_wk[i] - tilde_wk[i - 1]) / (k[1] - k[0])

    return vel[:-1]

group_vel_LP = np.zeros((Ncav - 1, 2), dtype = float)
group_vel_LP[:, 0] = omega_LP(wc, k_plot)[:-1]
group_vel_LP[:, 1] = vk_tilde(tilde_wk1, k_plot)

group_vel_UP = np.zeros((Ncav - 1, 2), dtype = float)
group_vel_UP[:, 0] = omega_UP(wc, k_plot)[:-1]
group_vel_UP[:, 1] = vk_tilde(tilde_wk2, k_plot)

np.savetxt("group_vel_LP.txt", group_vel_LP)
np.savetxt("group_vel_UP.txt", group_vel_UP)
