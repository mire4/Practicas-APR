import sys
import math
import numpy as np
import pickle
from sklearn import mixture

"""""
Descripcion de los valores que hay que pasar por el terminal.

   <dvdata>: Archivo con los datos
 <dvlabels>: Archivo con las etiquetas de los datos
        <k>: Numero de componentes por clase
    <model>: Despues de ejecutar mixgaussian-exp.py se nos habrá generado un modelo

Ejemplo: python mixgaussian-eva.py train-images-idx3-ubyte.pca20.npz train-labels-idx1-ubyte.pca20.npz 1 gmm.K1.rc0.1.mod

"""

if len(sys.argv) != 5:
  print('Usage: %s <dvdata> <dvlabels> <k> <model>' % sys.argv[0])
  sys.exit(1)

Xdv = np.load(sys.argv[1])['X']
xldv = np.load(sys.argv[2])['xl']
k = sys.argv[3]
fn = sys.argv[4]

# Normalizamos los datos
mu = np.mean(Xdv,axis=0)
sigma = np.std(Xdv,axis=0)
sigma[sigma == 0] = 1
Xdv = (Xdv - mu) / sigma

model = pickle.load(open(fn, 'rb'))

C = len(model)
M = Xdv.shape[0]
gdv = np.zeros((C,M))

for c, (pc, gmm) in enumerate(model):
  gdv[c] = math.log(pc) + gmm.score_samples(Xdv)

# Estimacion del error
labs = np.unique(xldv).astype(int)
idx = np.argmax(gdv, axis = 0)
edv = np.mean(np.not_equal(labs[idx], xldv)) * 100

# Formato para visualizarlo en el terminal
print('{:<8} {:<15}'.format('K', 'EDV'))

# Visualizacion en el terminal
print('{:<8} {:<15.2f}'.format(k, edv))