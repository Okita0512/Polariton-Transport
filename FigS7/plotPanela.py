#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 13:59:21 2023

@author: xchng
"""

import numpy as np
import matplotlib.pyplot as plt 
import HTCmodelPanela as model
import matplotlib as mpl

# Constants for conversion to au
au = 27.2114
fs2au = 41.341374575751
ang = 1.8897

# Load theory results
theory = np.loadtxt(f"group_vel_LP_lambda=24meV_3.txt")
theory_energy = theory[:,0] * au
theory_vel = theory[:,1] * fs2au / ang / 10.0 

# Calculates upper and lower polariton band
theta = model.θ
od_element = model.gc0*(np.cos(theta))**0.5
gc = model.gc*(model.NMol)**0.5
up = np.zeros(len(theta))
lp = np.zeros(len(theta))
omega_k = model.ωc 

ham_mat = np.zeros([2,2])

for i in range(len(theta)):
    ham_mat[0,0] = model.eExc
    ham_mat[1,1] = omega_k[i]

    up[i] = 0.5*(ham_mat[0,0]+ham_mat[1,1]) + 0.5*((ham_mat[0,0]-ham_mat[1,1])**2 + 4*gc[i]**2)**0.5
    lp[i] = 0.5*(ham_mat[0,0]+ham_mat[1,1]) - 0.5*((ham_mat[0,0]-ham_mat[1,1])**2 + 4*gc[i]**2)**0.5

# Computes lower polariton velocity from band
lpau = model.au*lp[211:]
kcheck = model.kx
dk = kcheck[1] - kcheck[0]
velocity = np.gradient(lp,dk)*model.fs2au/model.ang
vellp = velocity[211:]

# Plots velocity
plt.figure(figsize=(8.5,6.5),dpi=350)
mpl.rc('font',size=25.5)
plt.plot(lpau ,velocity[211:]/10,'k',linewidth = 4.1)
plt.plot(theory_energy,theory_vel,'g',linewidth = 4.1)
plt.plot(1.816,55.90,'go',fillstyle = 'none',markersize = 10.0,markeredgewidth=3.1)
plt.plot(1.80,56.53,'go',fillstyle = 'none',markersize = 10.0,markeredgewidth=3.1)
plt.plot(1.84,46.66,'go',fillstyle = 'none',markersize = 10.0,markeredgewidth=3.1)
plt.xticks(np.arange(1.74,1.86,0.02))
plt.xlim([1.73,1.865])
plt.ylim([0,70])
plt.xlabel('Energy (eV)')
plt.ylabel(r'$\tilde{v}_{g,-}$ ($\mu$m/ps)')
plt.savefig("polvelocity_24mev_18cavity.pdf",format='pdf',bbox_inches='tight')