#!/usr/bin/env python
import sys
import numpy as np
from sklearn import svm

if len(sys.argv) != 5:
  print('Usage: %s <tr.npz> <trl.npz> <dv.npz> <dvl.npz>' % sys.argv[0])
  sys.exit(1)

tr = np.load(sys.argv[1])
tr = tr[tr.files[0]]
trl = np.load(sys.argv[2])
trl = trl[trl.files[0]]
dv = np.load(sys.argv[3])
dv = dv[dv.files[0]]
dvl = np.load(sys.argv[4])
dvl = dvl[dvl.files[0]]

# normalizamos las características en [-1,1]
S = max(tr.max(), abs(tr.min())) 
tr /= S 
dv /= S

# probamos diferentes valores para el parámetro de penalización C, C>0,
# y hallamos el error en tr y dv para cada uno de ellos
for C in [1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4]:
  clf = svm.SVC(kernel = 'poly', C = C, degree=2).fit(tr, trl)
  etr = (trl != clf.predict(tr)).mean()
  edv = (dvl != clf.predict(dv)).mean()
  print("%8g %8.2f %8.2f" % (C, etr * 100, edv * 100))
