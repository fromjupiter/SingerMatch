import collections

import librosa
import os
import subprocess
import soundfile
import pandas as pd
import audio_utils
import numpy as np
from collections import defaultdict, OrderedDict
from itertools import repeat
import math
import json
import pickle


def flatten(l):
    for el in l:
        if isinstance(el, collections.Iterable) and not isinstance(el, (str, bytes)):
            yield from flatten(el)
        else:
            yield el


class Routines(object):
    def __init__(self, config):
        self.workspace = config['DEFAULT']['workspace']
        self.mfcc_dir = config['DEFAULT']['mfcc_dir']
        self.clipped_mfcc_dir = config['DEFAULT']['clipped_mfcc_dir']
        self.original_mp3_dir = config['DEFAULT']['original_mp3_dir']
        self.filtered_mp3_dir = config['DEFAULT']['filtered_mp3_dir']
        self.clipped_mp3_dir = config['DEFAULT']['clipped_mp3_dir']
        self.essentia_feature_dir = config['DEFAULT']['essentia_feature_dir']
        self.cqt_dir = config['DEFAULT']['cqt_dir']
        self.utils = audio_utils.AudioUtils()

    def to_spectrogram(self, skip=0):
        print("Generating spectrograms for clipped mp3s.")
        print("skipping {} lines".format(skip))
        count = 0
        with open(self.clipped_mp3_dir + '/all.list', 'r') as f:
            for l in f:
                count += 1
                if count <= skip:
                    continue
                l = l.strip()
                output_path = l.replace('mp3s-clipped', 'mp3s-cqt').replace('wav', 'png')
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                self.utils.save_spectrogram(l, output_path)
                print("Line {}: CQT generated to {}".format(count, output_path))

    def slice(self, skip=0):
        print("Slicing audios into 30s intervals")
        print("skipping {} lines".format(skip))
        count = 0
        with open(self.original_mp3_dir + '/all.list', 'r') as f1, \
                open(self.clipped_mp3_dir + '/all.list', 'a' if skip > 0 else 'w') as f2:
            for l in f1:
                count += 1
                if count <= skip:
                    continue
                l = '/' + l.strip()
                input_path = self.original_mp3_dir + l + '.mp3'
                output_prefix = self.clipped_mp3_dir + l
                os.makedirs(os.path.dirname(output_prefix), exist_ok=True)
                slices = self.utils.save_sliced_audio(input_path, output_prefix, 30)
                for sl in slices:
                    f2.write(sl + '\n')
                print("Line {}: Song {} has been sliced to {}".format(count, l, len(slices)))

    def filter(self, skip=0):
        print("skipping {} lines".format(skip))
        count = 0
        with open(self.clipped_mp3_dir + '/all.list', 'r') as f:
            for l in f:
                count += 1
                if count <= skip:
                    continue
                l = l.strip()
                input_path = l
                output_wav_path = l.replace(self.clipped_mp3_dir, self.filtered_mp3_dir)
                os.makedirs(os.path.dirname(output_wav_path), exist_ok=True)
                y, sr = librosa.load(input_path)
                y, sr = self.utils.load_vocal_audio(y, sr)
                soundfile.write(output_wav_path, y, sr)
                print("line {}, complete song {}".format(count, l))

    def split_list_filtered(self):
        with open(self.filtered_mp3_dir + '/all.list', 'r') as f, \
                open(self.filtered_mp3_dir + '/train.list', 'w') as trainf, \
                open(self.filtered_mp3_dir + '/test.list', 'w') as testf:
            albumsPerSinger = defaultdict(set)
            for l in f:
                singer, album = l.split('/')[:2]
                if 5 <= len(albumsPerSinger[singer]) and album not in albumsPerSinger[singer]:
                    # to test set
                    testf.write(l)
                else:
                    # to train set
                    albumsPerSinger[singer].add(album)
                    trainf.write(l)

    def split_list(self):
        with open(self.original_mp3_dir + '/all.list', 'r') as f, \
                open(self.original_mp3_dir + '/train.list', 'w') as trainf, \
                open(self.original_mp3_dir + '/test.list', 'w') as testf:
            albumsPerSinger = defaultdict(set)
            for l in f:
                singer, album = l.split('/')[:2]
                if 5 <= len(albumsPerSinger[singer]) and album not in albumsPerSinger[singer]:
                    # to test set
                    testf.write(l)
                else:
                    # to train set
                    albumsPerSinger[singer].add(album)
                    trainf.write(l)

    def gen_mfcc_train_set(self, nsamples: int, skip=0):
        print("skipping {} lines".format(skip))
        count = 0
        with open(self.original_mp3_dir + '/train.list', 'r') as f, \
                open(self.workspace + '/mfcc_train.data', 'a' if skip > 0 else 'w') as out:
            for l in f:
                count += 1
                if count <= skip:
                    continue
                lbl = l.split('/')[0]
                l = '/' + l.strip()
                htk_path = self.mfcc_dir + l + '.htk'
                data = self.utils.read_mfcc(htk_path)
                x = data[[int(x) for x in np.arange(0, len(data) - 1, (len(data) - 1) / nsamples)]]
                y = repeat(lbl, nsamples)
                for feats, label in zip(x, y):
                    out.write(label + ' ')
                    out.write(' '.join([str(round(i, 8)) for i in feats]) + '\n')
                print("line {}, sampled mfcc for song {}".format(count, l))

    def gen_mfcc_train_set_vocal(self, nsamples: int, skip=0):
        print("skipping {} lines".format(skip))
        count = 0
        with open(self.filtered_mp3_dir + '/train.list', 'r') as f, \
                open(self.workspace + '/vocal_mfcc_train.data', 'a' if skip > 0 else 'w') as out:
            for l in f:
                count += 1
                if count <= skip:
                    continue
                lbl = l.split('/')[0]
                l = '/' + l.strip()
                wav_path = self.filtered_mp3_dir + l + '.wav'
                y, sr = librosa.load(wav_path)
                hop_length = 512
                data = librosa.feature.mfcc(y=y, sr=sr, hop_length=hop_length).T
                # find start of audio
                start_of_song = 0
                sum_count = y[0]
                while sum_count == 0 and start_of_song < len(y):
                    sum_count += abs(y[start_of_song])
                    start_of_song += 1
                # find end of audio
                end_of_song = len(y)-1
                sum_count = y[end_of_song]
                while sum_count == 0 and end_of_song > 0:
                    sum_count += abs(y[end_of_song])
                    end_of_song -= 1
                if start_of_song >= len(y):
                    print("line {}, sampled nothing for song {}".format(count, l))
                    continue
                start_of_song = int(start_of_song/hop_length)
                end_of_song = int(end_of_song/hop_length)
                data = data[start_of_song:end_of_song]
                samples = [int(x) for x in np.arange(0, len(data) - 1, (len(data) - 1) / nsamples) if x < end_of_song]
                x = data[samples]
                y = y[samples]
                labels = repeat(lbl, nsamples)
                for feats, label in zip(x, labels):
                    out.write(label + ' ')
                    out.write(' '.join([str(round(i, 8)) for i in feats]) + '\n')
                print("line {}, sampled mfcc for song {}".format(count, l))

    def extract_essentia_features(self, skip=0):
        print("skipping {} lines".format(skip))
        count = 0
        cmd = self.workspace + "/streaming_extractor_music.exe {} {}"
        with open(self.original_mp3_dir + '/all.list', 'r') as f:
            for l in f:
                count += 1
                if count <= skip:
                    continue
                l = '/' + l.strip()
                input_path = self.original_mp3_dir + l + '.mp3'
                output_path = self.essentia_feature_dir + l + '.json'
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                subprocess.call(cmd.format(input_path, output_path))
                print("line {}, extracted essentia features for song {}".format(count, l))

    def gen_essentia_data_set(self):
        ohc = {'A': [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
               'A#': [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
               'B': [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
               'C': [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
               'C#': [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
               'D': [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
               'D#': [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
               'E': [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
               'F': [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
               'F#': [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
               'G': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
               'G#': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
               }
        for name in ('train', 'test'):
            res = []
            count = 0
            with open('{}/{}.list'.format(self.original_mp3_dir, name), 'r') as f:
                for l in f:
                    count += 1
                    l = '/' + l.strip()
                    json_path = self.essentia_feature_dir + l + '.json'
                    with open(json_path, 'r') as jf:
                        data = json.load(jf, object_pairs_hook=OrderedDict)
                        data.pop('metadata')
                        data['rhythm'].pop('beats_position')
                        data = pd.io.json.json_normalize(data, sep='_')
                    row = data.values.tolist()[0]
                    # remove last third and fourth attributes
                    row[-4] = row[-2]
                    row[-3] = row[-1]
                    row = row[:-2]
                    # scale: major or minor
                    row[-1] = 0 if row[-1] == 'minor' else 1
                    # one hot encode key
                    row[-2] = ohc.get(row[-2], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
                    res.append(list(flatten(row)))
                    print("line {}, completed song {}.".format(count, l))
            with open('{}/{}.data'.format(self.essentia_feature_dir, name), 'wb') as bf:
                pickle.dump(res, bf)
