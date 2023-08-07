import sys
import warnings
from sklearn.cluster import KMeans
import librosa
import numpy as np
import os
from tensorflow.keras.layers import GlobalAvgPool1D
if not sys.warnoptions:
    warnings.simplefilter("ignore")

input_length = 16000 * 30
import tensorflow as tf
n_mels = 128

# Definindo as configurações do MFCC
n_mfcc = 13
n_fft = 2048
hop_length = 512
num_segments = 10
n_clusters = 256

def pre_process_audio_mel_t(audio, sample_rate=16000):
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=sample_rate, n_mels=n_mels)
    mel_db = (librosa.power_to_db(mel_spec, ref=np.max) +40)/40
    min = np.min(mel_db)
    max = np.max(mel_db)
    mel_db = (mel_db - min) / (max - min)
    return mel_db.T

def plot_mel_spectrogram(audio, sample_rate=16000, n_mels=128):
    mel_db = pre_process_audio_mel_t(audio, sample_rate, n_mels)
    plt.figure(figsize=(10, 4))
    plt.imshow(mel_db.T, origin='lower', aspect='auto', cmap='viridis')
    plt.xlabel('Time')
    plt.ylabel('Mel Bin')
    plt.title('Mel Spectrogram')
    plt.colorbar(format='%+2.0f dB')
    plt.show()
# def preprocess_audio(signal,sr=16000):
#     # Carregando o arquivo de áudio e dividindo em segmentos
#     duration = signal.shape[0] / sr
#     samples_per_segment = int(sr * num_segments)
#     num_segments_per_audio = int(np.ceil(duration / num_segments))
#     remainder_samples = samples_per_segment * num_segments_per_audio - signal.shape[0]
#     signal = np.pad(signal, (0, remainder_samples))
#     segments = np.reshape(signal, (num_segments_per_audio, samples_per_segment))

#     # Extrai o MFCC de cada segmento
#     mfccs = []
#     for segment in segments:
#         mfcc = librosa.feature.mfcc(segment, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
#         # Normaliza o resultado para que os valores fiquem entre 0 e 1
#         mfcc = (mfcc - np.min(mfcc)) / (np.max(mfcc) - np.min(mfcc))
#         mfccs.append(mfcc.T)
#     mfccs = np.array(mfccs)
#     mfccs = mfccs.reshape(-1, tf.shape(mfccs)[2])
#     mfccs = KMeans(n_clusters=n_clusters).fit(mfccs)
#     return np.array(mfccs)

# def pre_process_audio_mel_t(audio, sample_rate=16000):
#     mel_spec = librosa.feature.melspectrogram(y=audio, sr=sample_rate, n_mels=n_mels)
#     mel_db = librosa.amplitude_to_db(mel_spec, ref=np.max)
#     mel_db = np.maximum(mel_db, mel_db.max() - 80)  # Remove noise floor

#     # Normalize to [0, 1]
#     mel_db /= 80.0
#     return mel_db.T

# def preprocess_audio(signal,sr=16000):
#     duration = signal.shape[0] / sr
#     samples_per_segment = int(sr * num_segments)
#     num_segments_per_audio = int(np.ceil(duration / num_segments))
#     remainder_samples = samples_per_segment * num_segments_per_audio - signal.shape[0]
#     signal = np.pad(signal, (0, remainder_samples))
#     segments = np.reshape(signal, (num_segments_per_audio, samples_per_segment))

#     # Extrai o MFCC de cada segmento
#     mfccs = []
#     for segment in segments:
#         mfcc = librosa.feature.mfcc(segment, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
#         mfccs.append(mfcc.T)
#     mfccs = np.abs(mfccs)
#     return np.array(mfccs)

def load_audio_file(file_path, input_length=input_length):
    try:
        data = librosa.core.load(file_path, sr=16000)[0]  # , sr=16000
    except ZeroDivisionError:
        data = []

    if len(data) > input_length:

        max_offset = len(data) - input_length

        offset = np.random.randint(max_offset)

        data = data[offset : (input_length + offset)]

    else:
        if input_length > len(data):
            max_offset = input_length - len(data)

            offset = np.random.randint(max_offset)
        else:
            offset = 0

        data = np.pad(data, (offset, input_length - len(data) - offset), "constant")

    # data = pre_process_audio_mel_t(data)
    return data


def random_crop(data, crop_size=128):
    start = np.random.randint(0, data.shape[0] - crop_size)
    # print('tempData',np.shape(data[start : (start + crop_size), :]))
    # data = np.reshape(data, (data.shape[0], -1))
    return data[start : (start + crop_size), :]
    # return data


def random_mask(data):
    new_data = data.copy()
    prev_zero = False
    for i in range(new_data.shape[0]):
        if np.random.uniform(0, 1) < 0.1 or (
            prev_zero and np.random.uniform(0, 1) < 0.5
        ):
            prev_zero = True
            new_data[i, :] = 0
        else:
            prev_zero = False

    return new_data


def save(path):
    data = pre_process_audio_mel_t(load_audio_file(path))
    np.save(path.replace(".mp3", ".npy"), data)
    return True

def rename_tracks(path,index,genre):
    newPath = path.replace(genre,str(index))
    os.rename(path,newPath)

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from tqdm import tqdm
    from glob import glob
    from multiprocessing import Pool

    base_path = "./audio"
    files = sorted(list(glob(base_path + "/*.mp3")))

    # print(np.shape(pre_process_audio_mel_t(load_audio_file('./Data/tracks/000000.wav'))))
    p = Pool(8)

    for i, _ in tqdm(enumerate(p.imap(save, files))):
        if i % 1000 == 0:
            print(i)

    # data = load_audio_file("/media/ml/data_ml/fma_medium/008/008081.mp3", input_length=16000 * 30)
    #
    # print(data.shape)
    #
    # new_data =random_mask(data)
    #
    # plt.figure()
    # plt.imshow(data.T)
    # plt.show()
    #
    # plt.figure()
    # plt.imshow(new_data.T)
    # plt.show()
    #
    # print(np.min(data), np.max(data))
