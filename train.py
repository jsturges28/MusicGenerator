import librosa
import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import pandas as pd
import librosa.display
import matplotlib.pyplot as plt
import os
import ffmpeg
from IPython.display import Audio

audio_file = os.path.abspath('C:/Users/stur8980/Downloads/melody-infraction-main-version-02-03-15043.mp3')

# Load the audio file
signal, sample_rate = librosa.load(audio_file, sr=None)

# Compute the Short-Time Fourier Transform (STFT)
stft = librosa.stft(signal, n_fft=2048, hop_length=512, win_length=2048, window='hann')

# Check the shape of the numpy array
print("Signal shape:", signal.shape)
