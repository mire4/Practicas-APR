import numpy as np
import matplotlib.pyplot as plt

# Cargamos los ficheros
dataExp = np.loadtxt('mixgaussian-exp.out')
dataEva = np.loadtxt('mixgaussian-eva.out')

fig, ax = plt.subplots()
ax.set_title('MNIST con 90% entrenamiento y 10% validación')
ax.grid()
ax.set_xlabel('Número de componentes')

ax.set_xlim([0.9,220])
ax.set_xscale('log')
ax.set_xticks([1,2,5,10,20,50,100,200])
ax.set_xticklabels([1,2,5,10,20,50,100,200])

ax.set_ylim([0,6])
ax.set_ylabel('Error de clasificación')
ax.set_yticks(range(7))

ax.plot(dataExp[:,0], dataExp[:,1], label = 'Entrenamiento', lw = 1, marker = '.', markersize = 5, color = 'b')
ax.plot(dataExp[:,0], dataExp[:,2], label = 'Test', lw = 1, marker = 'x', markersize = 5, color = 'k')
ax.plot(dataEva[:,0], dataEva[:,1], label = 'Eva', lw = 0, marker = 'd', markersize = 3, color = 'r')

ax.legend()

plt.savefig('mixgaussian-K.pdf');
plt.show()