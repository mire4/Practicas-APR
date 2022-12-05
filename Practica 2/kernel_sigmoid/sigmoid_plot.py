#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt

d = np.loadtxt('sigmoid.out')
fig, ax = plt.subplots()
ax.set_title('Kernel sigmoide en MNIST con tr 12k y dv 6k')
ax.grid()
ax.set_xscale('log')
ax.set_xlim([0.5e-3, 1.5e4])
ax.set_xlabel('C')
ax.set_ylim([15,70])
ax.set_ylabel('Error de clasificación')
ax.plot(d[:, 0], d[:, 1], label = 'tr', lw = 2, marker = 'o', markersize = 10)
ax.plot(d[:, 0], d[:, 2], label = 'dv', lw = 2, marker = 'x', markersize = 10)
ax.legend()
plt.savefig('sigmoid.pdf')
plt.show()