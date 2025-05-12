import numpy as np
import matplotlib.pyplot as plt 
import HTCmodelPanela as model
import time

# Reads data file
print("Loading data\n")
st = time.time()
ρData = np.loadtxt("Pii_population_panelb.txt")
ed = time.time()
print(f"Data loaded in {ed-st} seconds\n")

timeData = ρData[:,0]    # Extracts time data

# Extracts polaritonic wavepacket
stcalc = time.time()
ρiit = np.zeros((timeData.shape[0],model.NStates))
for tStep in range(timeData.shape[0]):
    ρiit[tStep,:] = ρData[tStep,1:]
edcalc = time.time()  
print(f"Data extracted in {edcalc-stcalc} seconds")
 
# Initializes array for polaritonic population
pop_ground = np.zeros((timeData.shape[0],))
pop_lp = np.zeros((timeData.shape[0],))
pop_dark = np.zeros((timeData.shape[0],))
pop_up = np.zeros((timeData.shape[0],))

# Assigns population into upper/lower polariton states and dark states
for i in range(0,timeData.shape[0]):
    ρiit_polbasis = ρiit[i,:]
    pop_ground[i] = ρiit_polbasis[0]
    pop_lp[i] = np.sum(ρiit_polbasis[1:model.NMod+1]) 
    pop_dark[i] = np.sum(ρiit_polbasis[model.NMod+1:model.NStates - model.NMod])
    pop_up[i] = np.sum(ρiit_polbasis[model.NStates - model.NMod:])

# Plots panel b
plt.figure(figsize =(8.5,6.5),dpi=350)
plt.rcParams.update({'font.size': 25.5})
plt.plot(timeData[0:250]*250.0/10,pop_lp[0:250],'r',linewidth=4.1)
plt.plot(timeData[0:250]*250.0/10,pop_up[0:250],'b',linewidth=4.1)
plt.plot(timeData[0:250]*250.0/10,pop_dark[0:250],'k',linewidth=4.1)
plt.xlabel('Time (fs)')
plt.ylabel('Population')
plt.xlim([0,200])
plt.legend(['LP','UP','Dark'],frameon=False,fontsize=23.5,bbox_to_anchor=(0.359,0.53))
plt.savefig("1_8eVCavity_1_816eVExcitation_24meVLambda.pdf",format='pdf',bbox_inches='tight')
