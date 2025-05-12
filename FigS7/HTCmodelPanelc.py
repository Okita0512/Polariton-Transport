import numpy as np
from numba import jit
import scipy.sparse as sp

@jit(nopython=True)
def bathParam(lamba, ωc,M, ndof):
    wmax = 10.0 * ωc

    c = np.zeros(( ndof ))
    ω = np.zeros(( ndof ))

    for d in range(ndof):
        ω[d] = ωc * np.tan((np.pi/2)*(1 - (d+1)/(ndof+1)  ))
        c[d] = np.sqrt(2*M*lamba/(ndof+1)) * ω[d]  
        
       
    return c, ω

fs2au = 41.341374575751
cminv2au = 4.55633*1e-6
eV2au = 0.036749405469679
K2au = 0.00000316678
amu = 1822.888
ps = 41341.374575751
ang = 1.8897
au = 27.2114
c = 137.0 

'''dtN -> nuclear time step (au)
    Nsteps -> number of nuclear time steps
    Total simulation time = Nsteps x dtN (au)'''
SimTime = 200             # in fs
dtN = 10.0               # in au originally 40
NSteps = int(SimTime/(dtN/fs2au)) + 1  

'''Esteps -> number of electronic time steps per nuclear time step
    dtE -> electronic time step (au)'''    
ESteps = 500   # originally 120              
dtE = dtN/ESteps     

NMol = 17651
NMod = 1071
NStates = NMol + NMod + 1                 # number of electronic states
M = 1                             # mass of nuclear particles (au)
NTraj = 1                        # number of trajectories
nskip = 10                          # save data every nskip steps of PLDM simulation

'''Bath parameters'''
NModes = 35 # Number of bath modes per sites
lambda_ = 0.024/au 
ωc = 0.0061992128657421585/au 
ck, ωk = bathParam(lambda_, ωc,M, NModes)
lambda_bath = 0.5*np.sum(ck**2/(M*ωk**2))
NR = NMol * NModes

eExc = 1.96 /au     # Exciton energy
gc0 =  0.120 / au / np.sqrt(NMol)   # light-matter coupling strength

Lx = 200*ang # lattice spacing
wc0 = 1.41/au  # cavity frequency
kz = wc0/c
# kx_max = NMod*2*np.pi/(NMol*Lx)
kx = 2 * np.pi *  np.arange(-NMod//2 + 1,NMod//2+1) / (NMol*Lx)
kx = kx[:NMod] 
nr = 1.0       # refractive index
ωc = (c/nr) * (kx**2.0 + kz**2.0)**0.5

θ = np.arctan(kx/kz)
gc = gc0 * (ωc/wc0)**0.5  * np.cos(θ)
xj = np.arange(0,NMol) * (Lx) # integer positions
