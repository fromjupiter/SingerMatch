from collections import defaultdict, OrderedDict
import pickle
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.metrics import f1_score
from routines import Routines
import configparser
import os

config = configparser.ConfigParser()
config.read('../../system.ini')
routines = Routines(config)


def train_GMM(dump_dir):
    print("Training GMM...")
    # Train model
    X = defaultdict(list)
    label_map = defaultdict(lambda: len(label_map))
    with open(routines.mfcc_dir+'/mfcc_train.data', 'r') as f:
        for l in f:
            l = l.strip().split(' ')
            label = l[0]
            label_map[label]
            X[label].append([float(x) for x in l[1:]])
    mods = OrderedDict()
    for label in label_map.keys():
        fn = dump_dir+label+'.obj'
        if os.path.exists(fn):
            continue
        mod = GaussianMixture(n_components=64, covariance_type='diag')
        mod.fit(X[label])
        print("Model trained successfully for artist {}! Dumping model to file..".format(label))
        with open(fn, 'wb') as df:
            pickle.dump(mod, df)
        mods[label] = mod

    return mods


models = OrderedDict()
mod_dir = routines.workspace+'/gmm_models/'
if os.path.exists(mod_dir) and os.listdir(mod_dir):
    for f in os.listdir(mod_dir):
        singer = f.split('.')[0]
        with open(mod_dir+f, 'rb') as rf:
            models[singer] = pickle.load(rf)
else:
    os.makedirs(mod_dir, exist_ok=True)
    models = train_GMM(mod_dir)

# Test model
pred = []
y_test = []
keys = list(models.keys())
with open(routines.original_mp3_dir+'/test.list', 'r') as f:
    for l in f:
        y_test.append(l.split('/')[0])
        l = '/' + l.strip()
        htk_path = routines.mfcc_dir + l + '.htk'
        data = routines.utils.read_mfcc(htk_path)
        likelihood = np.zeros(len(models))
        for i, mod in enumerate(models.values()):
            likelihood[i] = mod.score(data)
        res = np.argmax(likelihood)
        pred.append(keys[res])
        print("predicted *{}* for song: {}".format(keys[res], l))

y_test = np.array(y_test)
pred = np.array(pred)
accuracy = sum(y_test == pred)/len(y_test)
print("accuracy: {}".format(accuracy))
print("F1 score: {}".format(f1_score(y_test, pred, labels=keys, average='micro')))
# make confusion matrix
print(keys)
confusion_matrix = np.zeros((20, 20))
for i in range(0, len(y_test)):
    confusion_matrix[keys.index(y_test[i])][keys.index(pred[i])] += 1

for l in confusion_matrix.tolist():
    print(' '.join([str(int(x)) for x in l]))
