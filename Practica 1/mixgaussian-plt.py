import numpy as np
import matplotlib.pyplot as plt

data=np.loadtxt('mixgaussian-exp.out')
fig, ax = plt.subplots()
ax.set_title('MNIST con 90% entrenamiento y 10% validacion');
ax.grid();
ax.set_xlabel('Numero de componentes');
ax.set_xlim([0.9,220]);
ax.set_xscale('log');
ax.set_xticks([1,2,5,10,20,50,100,200])
ax.set_xticklabels([1,2,5,10,20,50,100,200])
ax.set_ylim([0,6]);
ax.set_ylabel('Error de clasificacion');
ax.set_yticks(range(7))
ax.plot(data[:,0], data[:,1], label = 'Entrenamiento', lw = 1, marker = 'o', markersize = 4)
ax.plot(data[:,0], data[:,2], label = 'Test', lw = 1, marker = 'x', markersize = 4)
ax.legend();
plt.savefig('mixgaussian-K.pdf');
plt.show();