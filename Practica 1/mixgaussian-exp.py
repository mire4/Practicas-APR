import sys
import math
import numpy as np
import pickle
from sklearn import mixture

"""""
Descripcion de los valores que hay que pasar por el terminal.

   <trdata>: Archivo con los datos
 <trlabels>: Archivo con las etiquetas de los datos
       <ks>: Lista con los diferentes componentes que queremos probar por clase
  <%%trper>: Porcentaje de datos para el entrenamiento
  <%%dvper>: Porcentaje de datos para el test

Ejemplo: python mixgaussian-exp.py train-images-idx3-ubyte.pca20.npz train-labels-idx1-ubyte.pca20.npz '1 2 5 10 20 50 100 200' 90 10

"""

if len(sys.argv) != 6:
  print('Usage: %s <trdata> <trlabels> <ks> <%%trper> <%%dvper>' % sys.argv[0])
  sys.exit(1)

X = np.load(sys.argv[1])['X']
xl = np.load(sys.argv[2])['xl']
trper = int(sys.argv[4])
dvper = int(sys.argv[5])

K = np.fromstring(sys.argv[3], dtype = int, sep = ' ')
rc = 0.1
seed = 23

N = X.shape[0]
np.random.seed(seed)
perm = np.random.permutation(N)
X = X[perm]
xl = xl[perm]

# Seleccionamos el conjunto de entrenamiento y test
Ntr = round(trper / 100 * N)
Xtr = X[:Ntr, :]
xltr = xl[:Ntr]
Ndv = round(dvper / 100 * N)
Xdv = X[N - Ndv:, :]
xldv = xl[N - Ndv:]

labs = np.unique(xltr).astype(int)
C = labs.shape[0]
N, D = Xtr.shape
M = Xdv.shape[0]
gtr = np.zeros((C, N))
gdv = np.zeros((C, M))

# Normalizamos los datos
mu = np.mean(Xtr, axis = 0)
sigma = np.std(Xtr, axis = 0)
sigma[sigma == 0] = 1
Xtr = (Xtr - mu) / sigma
Xdv = (Xdv - mu) / sigma

gmm = {}
idx = {}
etr = {}
edv = {}

# Abrimos un fichero de tipo .out para escribir el error del entrenamiento 
# y del test segun el numero de componentes usadas por clase.
file = open('mixgaussian-exp.out', 'w')

#Formato para verlo en el terminal
print('{:<8} {:<15} {:<10} {:<10}'.format('K','RC','ETR','EDV'))

for ind, value in enumerate(K):
  # print('Para ' + str(value) + ' componentes por clase')
  model = []
  for c, lab in enumerate(labs):
    # print('Etiqueta ' + str(lab))
    Xtrc = Xtr[xltr == lab]
    Nc = Xtrc.shape[0]
    pc = Nc / N
    gmm[ind] = mixture.GaussianMixture(n_components = value, reg_covar = rc, random_state = seed) #Creamos el modelo 
    gmm[ind].fit(Xtrc) #Entrenamos el modelo de mixturas 
    gtr[c] = math.log(pc) + gmm[ind].score_samples(Xtr) #score_samples nos devuelve el LL(0)
    gdv[c] = math.log(pc) + gmm[ind].score_samples(Xdv)
    model.append((pc, gmm[ind]))

  # Estimacion del error. Printear esto
  idx[ind] = np.argmax(gtr, axis = 0)
  etr[ind] = np.mean(np.not_equal(labs[idx[ind]], xltr)) * 100
  idx[ind] = np.argmax(gdv,axis = 0)
  edv[ind] = np.mean(np.not_equal(labs[idx[ind]], xldv)) * 100
  
  # print('Error del entrenamiento: ' + str(etr[ind]))
  # print('Error del test: ' + str(edv[ind]))

  file.write(str(K[ind])); file.write(str(' '))
  file.write(str(etr[ind])); file.write(str(' '))
  file.write(str(edv[ind])); file.write(str(' '))
  file.write('\n')

  # Visualizacion en el terminal
  print('{:<8} {:<15.1e} {:<10.2f} {:<10.2f}'.format(value, rc, (etr[ind]), (edv[ind])))

  # Con este comando codificamos en binario el modelo
  filename = 'gmm.K' + str(value) + '.rc0.1.mod'
  pickle.dump(model, open(filename, 'wb'))

# Cerramos el archivo .out
file.close()