#!/usr/bin/env python
import numpy as np
from sklearn import svm
import pickle

tr = np.load('train-images-idx3-ubyte.pca20.npz')  
tr = tr[tr.files[0]]
trl = np.load('train-labels-idx1-ubyte.pca20.npz') 
trl = trl[trl.files[0]]
S = max(tr.max(), abs(tr.min())) 
tr /= S
clf = pickle.load(open('rbf_save.clf', 'rb'))
etr = (trl != clf.predict(tr)).mean()
print("%8g %8.2f" % (clf.C, etr * 100))
pickle.dump(clf, open('rbf_save.clf', 'wb'))
