#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 13:59:21 2023

@author: xchng
"""

import numpy as np
import matplotlib.pyplot as plt 
import HTightBinding as model
import matplotlib as mpl


theory = np.loadtxt(f"group_vel_LP_lambda=12meV_3_v2.txt")
theory_energy = theory[:,0] * model.au
theory_vel = theory[:,1] *41.341374575751/1.8897/10

theta = model.θ
od_element = model.gc0*(np.cos(theta))**0.5
gc = model.gc*(model.NMol)**0.5
up = np.zeros(len(theta))
lp = np.zeros(len(theta))
omega_k = model.ωc 

ham_mat = np.zeros([2,2])

for i in range(len(theta)):
    ham_mat[0,0] = model.eExc + 2*model.tau
    ham_mat[1,1] = omega_k[i]

    up[i] = 0.5*(ham_mat[0,0]+ham_mat[1,1]) + 0.5*((ham_mat[0,0]-ham_mat[1,1])**2 + 4*gc[i]**2)**0.5
    lp[i] = 0.5*(ham_mat[0,0]+ham_mat[1,1]) - 0.5*((ham_mat[0,0]-ham_mat[1,1])**2 + 4*gc[i]**2)**0.5


lpau = model.au*lp[211:]

kcheck = model.kx
dk = kcheck[1] - kcheck[0]
test = np.gradient(lp,dk)*model.fs2au/model.ang

vellp = test[211:]
plt.figure(figsize=(8.5,6.5),dpi=350)
mpl.rc('font',size=25.5)
p1, = plt.plot(lpau ,test[211:]/10,'k',linewidth = 4.1)
p2, = plt.plot(theory_energy,theory_vel,'m',linewidth = 4.1)
p3, = plt.plot(theory_energy,theory_vel,'b',linewidth = 4.1)
p4, = plt.plot(theory_energy,theory_vel,'r',linewidth = 4.1)
p5, = plt.plot(theory_energy,theory_vel,'g',linewidth = 4.1)

plt.plot(1.80,58.94,'mo',fillstyle = 'none',markersize = 10.0,markeredgewidth=3.1)
plt.plot(1.816,60.5,'mo',fillstyle = 'none',markersize = 10.0,markeredgewidth=3.1)
plt.plot(1.84,56.86,'mo',fillstyle = 'none',markersize = 10.0,markeredgewidth=3.1)
plt.plot(1.80,58.31,'bo',fillstyle = 'none',markersize = 10.0,markeredgewidth=3.1)
plt.plot(1.816,61.61,'bo',fillstyle = 'none',markersize = 10.0,markeredgewidth=3.1)
plt.plot(1.84,56.85,'bo',fillstyle = 'none',markersize = 10.0,markeredgewidth=3.1)
plt.plot(1.80,58.15,'ro',fillstyle = 'none',markersize = 10.0,markeredgewidth=3.1)
plt.plot(1.816,61.21,'ro',fillstyle = 'none',markersize = 10.0,markeredgewidth=3.1)
plt.plot(1.84,56.97,'ro',fillstyle = 'none',markersize = 10.0,markeredgewidth=3.1)
plt.plot(1.80,57.97,'go',fillstyle = 'none',markersize = 10.0,markeredgewidth=3.1)
plt.plot(1.816,60.57,'go',fillstyle = 'none',markersize = 10.0,markeredgewidth=3.1)
plt.plot(1.84,55.77,'go',fillstyle = 'none',markersize = 10.0,markeredgewidth=3.1)
plt.xticks(np.arange(1.74,1.86,0.02))
plt.xlim([1.73,1.865])
plt.ylim([0,70])
# plt.plot(1.83,241.03,'ro')
plt.legend([p1,p2,p3,p4,p5],['No Bath','$\omega_f$ = 3.1 meV','$\omega_f$ = 6.2 meV','$\omega_f$ = 12.4 meV','$\omega_f$ = 18.6 meV'],frameon=False,
           fontsize=21)


plt.xlabel('Energy (eV)')
plt.ylabel(r'$\tilde{v}_{g,-}$ ($\mu$m/ps)')
plt.savefig("polvelocity_12mev_18cavity.pdf",format='pdf',bbox_inches='tight')
