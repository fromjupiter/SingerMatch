from collections import defaultdict, OrderedDict
import pickle
import re
import librosa
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.metrics import f1_score
from routines import Routines
import configparser
import os

config = configparser.ConfigParser()
config.read('../../system.ini')
routines = Routines(config)

window_size = None  # about 60s
sliding_hop = 640  # about 15s


def train_GMM(dump_dir):
    print("Training GMM...")
    # Train model
    X = defaultdict(list)
    label_map = defaultdict(lambda: len(label_map))
    with open(routines.mfcc_dir+'/vocal_mfcc_train.data', 'r') as f:
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
mod_dir = routines.workspace+'/vocal_gmm_models/'
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
with open(routines.filtered_mp3_dir+'/test.list', 'r') as f:
    prev_label = None
    prev_song = None
    prev_data = None
    for l in f:
        l = l.strip()
        label = l.split('/')[0]
        song = l.split('/')[-1][:-2]
        song = re.sub('_[0-9]*$', '', l.split('/')[-1])

        wav_path = routines.filtered_mp3_dir + '/' + l + '.wav'
        y, sr = librosa.load(wav_path)
        data = librosa.feature.mfcc(y=y, sr=sr).T
        if song != prev_song or label != prev_label:
            if prev_data is not None:
                if window_size is None:
                    # predict whole song
                    y_test.append(prev_label)
                    likelihood = np.zeros(len(models))
                    for i, mod in enumerate(models.values()):
                        likelihood[i] = mod.score(prev_data)
                    res = np.argmax(likelihood)
                    pred.append(keys[res])
                    print("predicted *{}* for song: |{}|-{}".format(keys[res], prev_label, prev_song))
                else:
                    # predict sliding window
                    start = 0
                    while start+sliding_hop < len(prev_data):
                        y_test.append(prev_label)
                        sliced_data = prev_data[start:start+sliding_hop]
                        likelihood = np.zeros(len(models))
                        for i, mod in enumerate(models.values()):
                            likelihood[i] = mod.score(sliced_data)
                        res = np.argmax(likelihood)
                        pred.append(keys[res])
                        print("predicted *{}* for song: |{}|-{}-slice{}".format(keys[res], prev_label, prev_song, start))
                        start += sliding_hop
            # set prev
            prev_label = label
            prev_song = song
            prev_data = data
        else:
            prev_data = np.concatenate((prev_data, data))

    # predict last song
    y_test.append(prev_label)
    likelihood = np.zeros(len(models))
    for i, mod in enumerate(models.values()):
        likelihood[i] = mod.score(prev_data)
    res = np.argmax(likelihood)
    pred.append(keys[res])
    print("predicted *{}* for song: |{}|-{}".format(keys[res], prev_label, prev_song))


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
