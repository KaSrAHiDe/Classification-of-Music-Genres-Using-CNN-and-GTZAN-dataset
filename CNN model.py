# IMPORTING LIBRARIES
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy
import os
import pickle
import librosa  # Best py_package for music and audio analysis.
import librosa.display
from IPython.display import Audio  # Used for playing the audio in the jupyterlab.
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow import keras

# Reading the data from the dataset by setting the path
"""
**GTZAN Dataset**
source of audio data is the GTZAN dataset,
which consists of 3sec segments of music from 10 different genres,
each having 100 recordings (for a total of 1000).
Genres: Blues, Classical, Country, Disco, Hiphop, Jazz, Metal, Pop, Reggae, Rock
"""
df = pd.read_csv("Add your features_3_sec.csv directory here...")
df.head()

# Data preprocessing
df.shape
df.dtypes

# Dropping the unwanted labels
df = df.drop(labels="filename", axis=1)

# Understanding the Audio Files
audio_recording = "Add your Data\\genres_original\\classical\\classical.00000.wav directory..."
data, sr = librosa.load(audio_recording)
print(type(data), type(sr))
librosa.load(audio_recording, sr=45600)

# Play sample file
# IPython.display.Audio(audio_data, rate=sr)

"""
data, sr = librosa.load(audio_recording) It loads and decodes the audio as a time series y.
sr = sampling rate of y. It is the number of samples per second.
20 kHz is the audible range for human beings. So it is used as the default value for sr.
In this code I'm using sr as 45600Hz.
"""

# Spectrogram visualization
"""
A spectrogram is a visual way of representing
the signal loudness of a signal over time
at various frequencies present in a particular waveform.
"""
stft = librosa.stft(data)
stft_mag = np.abs(stft)
plt.figure(figsize=(14, 6))
librosa.display.specshow(
    librosa.amplitude_to_db(stft_mag), sr=sr, x_axis="time", y_axis="hz"
)
plt.colorbar()

"""
**Spectral Rolloff**
is the frequency below which a specified percentage of the total spectral energy, e.g. 85%
librosa.feature.spectral_rolloff computes the rolloff frequency for each frame in a signal.
"""
from sklearn.preprocessing import normalize

spectral_rolloff = librosa.feature.spectral_rolloff(y=data, sr=sr)[0]
plt.figure(figsize=(12, 4))
librosa.display.waveshow(data, sr=sr, alpha=0.4, color="#2B4F72")
"""
**Chroma Feature**
It is a powerful tool for analyzing music features whose pitches
can be meaningfully categorized and whose tuning approximates to the equal-tempered scale.
One main property of chroma features is that they capture harmonic and melodic characteristics
of music while being robust to changes in timbre and instrumentation.
"""
import librosa.display as lplt

chroma = librosa.feature.chroma_stft(y=data, sr=sr)
plt.figure(figsize=(16, 6))
lplt.specshow(chroma, sr=sr, x_axis="time", y_axis="chroma", cmap="coolwarm")
plt.colorbar()
plt.title("chroma Features")
plt.show()
"""
**Zero Crossing**
Is a measure of the number of times in a given time interval/frame that the amplitude
of the speech signals passes through a value of zero.
"""
start = 1000
end = 1200
plt.figure(figsize=(14, 5))
plt.plot(data[start:end], color="#2B4F72")
plt.grid()

zero_cross_rate = librosa.zero_crossings(data[start:end], pad=False)
print("The number of zero-crossings is:", sum(zero_cross_rate))

# Feature Extraction
"""
On the last line is ‘label’ and will encode it with the function LabelEncoder() of sklearn.preprocessing.
"""
class_list = df.iloc[:, -1]
convertor = LabelEncoder()
"""
fit_transform(): Fit label encoder and return encoded labels.
"""
y = convertor.fit_transform(class_list)
y
print(df.iloc[:, :-1])

# Scaling the Features
"""
Standard scaler is used to standardize features by removing the mean and scaling to unit variance.
"""
from sklearn.preprocessing import StandardScaler

fit = StandardScaler()
X = fit.fit_transform(np.array(df.iloc[:, :-1], dtype=float))

# Dividing Data Into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
len(y_train)
len(y_test)

# Building the Model
"""
The last part, The features have been extracted from the raw data and now we have to train the model.
I will be using CNN(Convolutional Neural Networks)Algorithm for training our model.
I choose this approach because various forms of research show it to have the best results for this project.
"""
from keras.models import Sequential


def trainModel(model, epochs, optimizer):
    batch_size = 128
    model.compile(
        optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics="accuracy"
    )
    return model.fit(
        X_train,
        y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
    )


def plotValidate(history):
    print("Validation Accuracy", max(history.history["val_accuracy"]))
    pd.Dataframe(history.history).plot(figsize=(12, 6))
    plt.show()


import keras as k

model = k.models.Sequential(
    [
        k.layers.Dense(512, activation="relu", input_shape=(X_train.shape[1],)),
        k.layers.Dropout(0.2),
        k.layers.Dense(256, activation="relu"),
        k.layers.Dropout(0.2),
        k.layers.Dense(128, activation="relu"),
        k.layers.Dropout(0.2),
        k.layers.Dense(64, activation="relu"),
        k.layers.Dropout(0.2),
        k.layers.Dense(32, activation="softmax"),
    ]
)
print(model.summary())
model_history = trainModel(model=model, epochs=550, optimizer="adam")
"""
For the CNN model, I had used the Adam optimizer(it gave me the best results after evaluating other optimizers)for training the model.
The epoch that was chosen for the training model is 550.
Dropout is used to prevent overfitting.
The model accuracy can be increased by further increasing the epochs.
"""

# Model Evaluation
test_loss, test_acc = model.evaluate(X_test, y_test, batch_size=128)
print("The test Loss is:", test_loss)
print("\nThe Best test Accuracy is:", test_acc * 100)

# Save the model
model.save("Zirak")
