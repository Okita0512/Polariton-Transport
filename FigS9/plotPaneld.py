import numpy as np
import matplotlib.pyplot as plt 
import time

# Load wavepacket data
print("Loading data\n")
st = time.time()
ρData = np.loadtxt("Pii_population_paneld.txt")
ed = time.time()
print(f"Data loaded in {ed-st} seconds\n")

# Initialize model parameters used for plots
NMol = 20001
NMod = 281
NStates = NMol + NMod + 1

timeData = ρData[:,0]           # Extracts time data

# Extracts polaritonic coefficients of the wavepacket
stcalc = time.time()
ρiit = np.zeros((timeData.shape[0],NStates))
for tStep in range(timeData.shape[0]):
    ρiit[tStep,:] = ρData[tStep,1:]
edcalc = time.time()  
print(f"Data extracted in {edcalc-stcalc} seconds")
 
# Initializes array to store polariton population
pop_ground = np.zeros((timeData.shape[0],))
pop_lp = np.zeros((timeData.shape[0],))
pop_dark = np.zeros((timeData.shape[0],))
pop_up = np.zeros((timeData.shape[0],))

# Assigns population into upper/lower polariton states and dark states
for i in range(0,timeData.shape[0]):
    ρiit_polbasis = ρiit[i,:]
    pop_ground[i] = ρiit_polbasis[0]
    pop_lp[i] = np.sum(ρiit_polbasis[1:NMod+1]) 
    pop_dark[i] = np.sum(ρiit_polbasis[NMod+1:NStates - NMod])
    pop_up[i] = np.sum(ρiit_polbasis[NStates - NMod:])

# Plots panel d
plt.figure(figsize =(8.5,6.5),dpi=350)
plt.rcParams.update({'font.size': 25.5})
plt.plot(timeData*200.0/8,pop_lp,'r',linewidth=4.1)
plt.plot(timeData*200.0/8,pop_up,'b',linewidth=4.1)
plt.plot(timeData*200.0/8,pop_dark,'k',linewidth=4.1)
plt.xlabel('Time (fs)')
plt.ylabel('Population')
plt.xlim([0,200])
plt.savefig("paneld.pdf",format='pdf',bbox_inches='tight')
