import librosa
import librosa.display
import soundfile
import vamp
import math
from matplotlib import pyplot as plt
import numpy as np
from ext.HTK import HTKFile

class AudioUtils(object):
    def __init__(self):
        pass

    def save_spectrogram(self, mp3_path: str, img_path: str):
        y, sr = librosa.load(mp3_path,sr=44100)
        cqt = librosa.amplitude_to_db(np.abs(librosa.cqt(y, sr=sr)), ref=np.max)
        plt.figure(figsize=(6, 4))
        librosa.display.specshow(cqt)
        plt.axis('off')
        plt.savefig(img_path, bbox_inches='tight', pad_inches=0)
        plt.close()

    #
    # extract vocal from audio data
    # https://librosa.github.io/librosa_gallery/auto_examples/plot_vocal_separation.html
    #
    def load_vocal_audio(self, y, sr):
        S_full, phase = librosa.magphase(librosa.stft(y))
        S_filter = librosa.decompose.nn_filter(S_full,
                                               aggregate=np.median,
                                               metric='cosine',
                                               width=int(librosa.time_to_frames(2, sr=sr)))
        S_filter = np.minimum(S_full, S_filter)
        margin_i, margin_v = 2, 10
        power = 2
        mask_v = librosa.util.softmask(S_full - S_filter,
                                       margin_v * S_filter,
                                       power=power)
        S_foreground = mask_v * S_full
        output_data = librosa.griffinlim(S_foreground)
        return output_data, sr

    def melodia_filter(self, y, sr, smooth=0.25):
        hop, data = vamp.collect(y, sr, 'mtg-melodia:melodia', parameters={"minfqr": 49, "maxfqr": 3200, "voicing": 3})['vector']
        hop = 128 / 44100.0
        # filter non-melody frames
        new_y = []
        dist = 0
        for i, s in enumerate(y):
            if data[int(max(i / sr - 8, 0) / hop)] > 0:
                dist = 0
                new_y.append(s)
            elif dist < smooth:
                dist += 1/44100.0
                new_y.append(s)
        return np.array(new_y), sr

    def save_sliced_audio(self, input_path: str, output_prefix: str, interval=30):
        y, sr = librosa.load(input_path)
        y, sr = self.melodia_filter(y, sr)
        size = math.ceil(len(y) / (interval * sr))
        y = np.pad(y, (0, size*interval*sr-len(y)), mode='constant', constant_values=0.0)
        y_list = np.array_split(y, size)
        res = []
        for i, yc in enumerate(y_list):
            fp = '{}_{}.wav'.format(output_prefix, i)
            soundfile.write(fp, yc, sr)
            res.append(fp)
        return res

    def read_mfcc(self, htk_path: str):
        htk_reader = HTKFile()
        htk_reader.load(htk_path)
        result = np.array(htk_reader.data)
        return result
