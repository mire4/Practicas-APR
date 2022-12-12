#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt

d = np.loadtxt('polyG10Coef1.out')

fig, ax = plt.subplots()
ax.set_title('Kernel polinomico en MNIST con tr 12k y dv 6k')
ax.grid()

ax.set_xscale('log')
ax.set_xlim([0.5e-3, 1.5e4])
ax.set_xlabel('C')

ax.set_ylim([-0.5,6])
ax.set_ylabel('Error de clasificación')

ax.plot(d[:, 0], d[:, 1], label = 'Entrenamiento', lw = 1, marker = 'o', markersize = 5, color = 'b')
ax.plot(d[:, 0], d[:, 2], label = 'Test', lw = 1, marker = 'x', markersize = 5, color = 'k')

ax.legend()

plt.savefig('polyG10Coef1.png')
plt.show()