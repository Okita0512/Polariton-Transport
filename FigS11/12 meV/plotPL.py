import numpy as np
import matplotlib.pyplot as plt 
import time


plSpect = np.zeros((1000,41))

for i in range(41):
    signal = np.loadtxt('signal_'+str(i)+'.txt')
    signal /= np.max(signal)
    plSpect[:,i] = signal
plt.figure(figsize=(8.9,5.9),dpi=400)
plt.imshow(1-plSpect,aspect='auto',origin='lower',extent=[0,40,1.7,2.6],cmap='gray',vmin=-0.05,vmax=1.5)
plt.ylim([1.7,2.4])
plt.xlabel(r'$k$ $(2\pi/L)$',fontsize=24)
plt.xticks([0,10,20,30,40],fontsize=24)
plt.ylabel('Energy (eV)',fontsize=24)
plt.yticks([1.8,2.0,2.2,2.4],fontsize = 24)
cbar = plt.colorbar()
cbar.set_label('Reflectance', fontsize=24)
cbar.ax.tick_params(labelsize=24)
plt.savefig("plSpectra_12meV.svg",format='svg',bbox_inches='tight')
 