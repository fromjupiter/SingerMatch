import configparser
import pickle
from sklearn.preprocessing import normalize

import numpy as np
from sklearn.metrics import f1_score

from routines import Routines
import xgboost as xgb

config = configparser.ConfigParser()
config.read('../../system.ini')
routines = Routines(config)

with open(routines.essentia_feature_dir+'/train.data', 'rb') as f:
    Xtrain = pickle.load(f)

melo_train = []
with open(routines.essentia_feature_dir+'/trainMelody.txt', 'r') as f:
    for l in f:
        melo_train.append([float(x) for x in l.strip().split(',')])
melo_train = normalize(melo_train, axis=1, norm='l1')[:, :128]
Xtrain = np.concatenate((melo_train, Xtrain), axis=1)

ytrain = []
with open(routines.original_mp3_dir + '/train.list', 'r') as f:
    for l in f:
        ytrain.append(l.strip().split('/')[0])

with open(routines.essentia_feature_dir+'/test.data', 'rb') as f:
    Xtest = pickle.load(f)

melo_test = []
with open(routines.essentia_feature_dir+'/testMelody.txt', 'r') as f:
    for l in f:
        melo_test.append([float(x) for x in l.strip().split(',')])
melo_test = normalize(melo_test, axis=1, norm='l1')[:, :128]
Xtest = np.concatenate((melo_test, Xtest), axis=1)

keys = []
ytest = []
with open(routines.original_mp3_dir + '/test.list', 'r') as f:
    for l in f:
        label = l.strip().split('/')[0]
        ytest.append(label)
        if len(keys) == 0 or keys[-1] != label:
            keys.append(label)

Xtrain = np.array(Xtrain)
ytrain = np.array(ytrain)
Xtest = np.array(Xtest)
ytest = np.array(ytest)

model = xgb.XGBClassifier(random_state=1, max_depth=3, n_estimators=1000)
print("Start training XGBoost classifier!!!")
model.fit(Xtrain, ytrain)

pred = model.predict(Xtest)
accuracy = sum(ytest == pred)/len(ytest)
print("accuracy: {}".format(accuracy))
print("F1 score: {}".format(f1_score(ytest, pred, labels=keys, average='micro')))
# make confusion matrix
print(keys)
confusion_matrix = np.zeros((20, 20))
for i in range(0, len(ytest)):
    confusion_matrix[keys.index(ytest[i])][keys.index(pred[i])] += 1

for l in confusion_matrix.tolist():
    print(' '.join([str(int(x)) for x in l]))
