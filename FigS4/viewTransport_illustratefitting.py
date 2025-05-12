import numpy as np
import matplotlib.pyplot as plt 
import time
import matplotlib as mpl

# Load wavepacket data
print("Loading data\n")
st = time.time()
ρData = np.loadtxt("Pii_wavepacket.txt")
ed = time.time()
print(f"Data loaded in {ed-st} seconds\n")

ang = 1.8897 # Converts angstrom to au

# Initialize model parameters used for plots
NMol = 20001
Lx = 40

timeData = ρData[:,0]     # Extracts time data

# Extracts spatial component of the wavepacket
stcalc = time.time()
ρiit = np.zeros((timeData.shape[0],NMol))
for tStep in range(timeData.shape[0]):
    ρiit[tStep,:] = ρData[tStep,2:NMol + 2]
edcalc = time.time()  
print(f"Data extracted in {edcalc-stcalc} seconds")
 
# Rescale time axis so that final time is 200 fs as in simulation 
timeStep = 200/8.0

# Plots panel (a)
color = ['r','g','b','m','c']
init = 0
plt.figure(figsize=(7.5,5),dpi=350)
mpl.rc('font',size=20.5)
for ts in range(61,82,5):
    time = round(timeData[ts] * timeStep)
    plt.plot(np.arange(NMol)*Lx/ang/1e4,ρiit[ts,:]*1e3,label=f"{time} fs",color = color[init])
    init+=1
plt.ylabel(r'$\tilde{\rho}_n$ (x$10^{-3}$)')
plt.xlabel(r'$\tilde{X}$ (x$10^3$ nm)')
plt.legend(frameon=False,fontsize =17.5)
plt.savefig("Piit_snapshot_illustrate_photmatt.pdf",format='pdf',bbox_inches='tight')

# Extracts wavefront of wavepacket
wFront = np.zeros(timeData.shape[0])
for i in range(0,timeData.shape[0]):
    sense = 0
    for j in range(0,NMol):
        sense += ρiit[i,j]/np.sum(ρiit[i,:])
        if sense >= 0.029:
            wFront[i] = (j)*Lx
            break
wFront = wFront/ ang
coef = np.polyfit(timeData[61:]*timeStep,wFront[61:],1)

# Plots panel (b)
poly1d_fn = np.poly1d(coef)
plt.figure(figsize=(7.5,5),dpi=350)
mpl.rc('font',size=20.5)
plt.plot(timeData[61:]*timeStep,wFront[61:]/1e4,'ro', fillstyle='none',markersize=8)
plt.plot(timeData[61:]*timeStep,poly1d_fn(timeData[61:]*timeStep)/1e4,'--k',linewidth=2.1)
plt.xlabel('Time (fs)')
plt.ylabel(r'$\tilde{X}$ (x$10^3$ nm)')
plt.savefig("velocity_fit_photmatt.pdf",format='pdf',bbox_inches='tight')