#!/usr/bin/env python
import numpy as np
from sklearn import svm
import pickle

tr=np.load('train-images-idx3-ubyte.pca20.npz');  tr=tr[tr.files[0]];
trl=np.load('train-labels-idx1-ubyte.pca20.npz'); trl=trl[trl.files[0]];
S=max(tr.max(),abs(tr.min())); tr/=S;
C=0.1;
clf=svm.SVC(kernel='poly',C=C,degree=3,gamma=10,coef0=1).fit(tr,trl);
etr=(trl!=clf.predict(tr)).mean();
print("%8g %8.2f" % (C,etr*100));
pickle.dump(clf,open('poly_save.clf','wb'));
