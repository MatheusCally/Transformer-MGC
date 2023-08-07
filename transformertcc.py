import numpy as np
import tensorflow as tf
import librosa
import time
from prepare_data import get_id_from_path, random_crop, labels_to_vector
from tensorflow.keras.layers import GlobalAvgPool1D
import json
from glob import glob
from prepare_data import get_id_from_path
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
num_layers = 2
d_model = 128
dff = 512
num_heads = 1
dropout_rate = 0.1
input_length = 1000
output_length = 10
n_mels = 128
EPOCHS = 10
BUFFER_SIZE = 20000
BATCH_SIZE = 8

def make_batches(ds):
  return (
      ds
      .cache()
      .shuffle(BUFFER_SIZE)
      .batch(BATCH_SIZE)
      .prefetch(tf.data.experimental.AUTOTUNE))

def data_generation_custom(list,class_mapping):
    paths, labels = zip(*list)
    labels = [labels_to_vector(x, class_mapping) for x in labels]
    crop_size = np.random.randint(128, 256)
    # X = np.array([random_crop(np.load(x), crop_size=crop_size) for x in paths])
    X = np.array([np.load(x) for x in paths])
    # X = GlobalAvgPool1D()(X)
    Y = np.array(labels)
    return X, Y

def returnModel():
    inputs = tf.keras.layers.Input(shape=(938,128))
    x = tf.keras.layers.Dense(256, activation='relu')(inputs)
    for i in range(num_layers):
      # Multi-Head Attention
      attention = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=256)(x, x)
      attention = tf.keras.layers.Dropout(0.1)(attention)
      add = tf.keras.layers.Add()([attention, x])
      # Feedforward
      feedforward = tf.keras.layers.Dense(dff, activation='relu')(add)
      feedforward = tf.keras.layers.Dense(256)(feedforward)
      feedforward = tf.keras.layers.Dropout(0.1)(feedforward)
      x = tf.keras.layers.Add()([feedforward, add])

    pooling = tf.keras.layers.GlobalAveragePooling1D()(x)
    outputs = tf.keras.layers.Dense(output_length, activation='softmax')(pooling)
    return tf.keras.Model(inputs=inputs, outputs=outputs)

def predictAndCount(model,genreIndex):
    predicts = []
    for i in range(0,100):
      if not (i == 54 and genreIndex == 5):
        if i < 10:
          input = np.array([np.load('./Data/tracks/' + str(genreIndex) + '0000' + str(i) + '.npy')])
        else:
          input = np.array([np.load('./Data/tracks/' + str(genreIndex) +'000' + str(i) + '.npy')])
        tensor = tf.convert_to_tensor(input)
        predict = model.predict(tensor)
        predicts.append(tf.argmax(predict,axis=-1))
      # print(tf.argmax(predict,axis=-1))
    percentage = predicts.count([genreIndex])/100
    print('Percentage for genre ' + str(genreIndex) + ': ',percentage)
    return percentage

def gptTransformer():
    CLASS_MAPPING, samples = classesAndSamples()
    train, val = train_test_split(
        samples, test_size=0.1, random_state=1337
    )
    datasetTrain = make_batches(tf.data.Dataset.from_tensor_slices(data_generation_custom(train,CLASS_MAPPING)))
    datasetVal = make_batches(tf.data.Dataset.from_tensor_slices(data_generation_custom(val,CLASS_MAPPING)))
    # model = returnModel()
    # model.load_weights('./newCheckpoints/tccCheckpoint-bkp.h5')
    model = tf.keras.models.load_model('./newCheckpoints/tccCheckpoint90.h5')
    model.summary()
    input = np.array([np.load('./audio/betovenClassical.npy')])
    tensor = tf.convert_to_tensor(input)
    predict = model.predict(tensor)
    print(tf.argmax(predict,axis=-1))
    percentages = {}
    for i in [0,1,4,5,6]:
      percentages[i] = predictAndCount(model,i)
    print(percentages)




    # checkpointRestore = tf.train.Checkpoint(model=model)

    # # Restaurar os pesos do modelo
    # checkpointRestore.restore(tf.train.latest_checkpoint("./newCheckpoints/tccCheckpoint.h5"))


    # checkpoint = ModelCheckpoint(
    #     "./newCheckpoints/tccCheckpoint.h5",
    #     monitor="val_accuracy",
    #     verbose=1,
    #     save_best_only=True,
    #     mode="max"
    # )
    # model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # model.summary()
    # model.fit(datasetTrain, epochs=200, validation_data=datasetVal, callbacks=[checkpoint])

def classesAndSamples():
    CLASS_MAPPING = json.load(open("./Data/mappingGtzan.json"))
    id_to_genres = json.load(open("./Data/tracks_genre.json"))
    id_to_genres = {int(k): v for k, v in id_to_genres.items()}

    base_path = "./Data"
    files = sorted(list(glob(base_path + "/tracks/*.npy")))
    files = filter(lambda file: int(file[-10]) in [0,1,4,5,6],files)
    files = [x for x in files if id_to_genres[int(get_id_from_path(x))]]
    labels = [id_to_genres[int(get_id_from_path(x))] for x in files]
    samples = list(zip(files, labels))
    return CLASS_MAPPING,samples

def treatData():
    CLASS_MAPPING = json.load(open("./Data/mappingGtzan.json"))
    id_to_genres = json.load(open("./Data/tracks_genre.json"))
    id_to_genres = {int(k): v for k, v in id_to_genres.items()}

    base_path = "./Data"
    files = sorted(list(glob(base_path + "/tracks/*.npy")))
    files = filter(lambda file: int(file[-10]) in [0,1,4,5,6],files)
    files = [x for x in files if id_to_genres[int(get_id_from_path(x))]]
    labels = [id_to_genres[int(get_id_from_path(x))] for x in files]
    samples = list(zip(files, labels))
    print(samples)
    # for key in id_to_genres.keys():

def plot_mel_spectrogram(spectrogram):
    mel_db = spectrogram
    plt.figure(figsize=(10, 4))
    plt.imshow(mel_db.T, origin='lower', aspect='auto', cmap='viridis')
    plt.xlabel('Time')
    plt.ylabel('Mel Bin')
    plt.title('Mel Spectrogram')
    plt.colorbar(format='%+2.0f dB')
    plt.show()

def plot_waveform(audio, sample_rate=16000):
    plt.figure(figsize=(10, 4))
    plt.plot(audio)
    plt.xlabel('Amostras')
    plt.ylabel('Amplitude')
    plt.title('Audio Waveform')
    plt.show()

if __name__ == "__main__":
    # predict()
    gptTransformer()
    # treatData()
    
    
    

