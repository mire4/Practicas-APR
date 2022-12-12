#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt

d = np.loadtxt('linear.out')

fig, ax = plt.subplots()
ax.set_title('Kernel lineal en MNIST con tr 12k y dv 6k')
ax.grid()

ax.set_xscale('log')
ax.set_xlim([1e-2, 1.5e4])
ax.set_xlabel('C')

ax.set_ylim([8,15])
ax.set_ylabel('Error de clasificación')

ax.plot(d[:, 0], d[:, 1], label = 'Entrenamiento', lw = 1, marker = 'o', markersize = 5, color = 'b')
ax.plot(d[:, 0], d[:, 2], label = 'Test', lw = 1, marker = 'x', markersize = 5, color = 'k')

ax.legend()

plt.savefig('linear.png')
plt.show()