import tensorflow
from tensorflow import keras
from keras.datasets import mnist
from autoencoder import Autoencoder
from vae import VAE
from unet import UNET
import os
import numpy as np

LEARNING_RATE = 0.0005
BATCH_SIZE = 16
EPOCHS = 2

#SPECTROGRAMS_PATH = os.path.abspath("C:/Users/stur8980/Documents/GitHub/MusicGenerator/spectrograms/")
SPECTROGRAMS_PATH = os.path.abspath("./spectrograms/")

def load_mnist():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.astype("float32") / 255 # normalize between 0 and 1
    x_train = x_train.reshape(x_train.shape + (1,))

    x_test = x_test.astype("float32") / 255 # normalize between 0 and 1
    x_test = x_test.reshape(x_test.shape + (1,))

    return x_train, y_train, x_test, y_test

def load_fsdd(spectrograms_path):
    x_train = []
    for root, _, file_names in os.walk(spectrograms_path):
        for file_name in file_names:
            file_path = os.path.join(root, file_name)
            spectrogram = np.load(file_path) # (n_bins, n_frames)
            x_train.append(spectrogram)

    x_train = np.array(x_train)
    x_train = x_train[..., np.newaxis] # -> (3000, 256, 64, 1) (num_samples, n_bins, n_frames, 1)
    return x_train




def train_vae(x_train, learning_rate, batch_size, epochs):
    autoencoder = VAE(input_shape=[256,64,1],
                              conv_filters=[512, 256, 128, 64, 32],
                              conv_kernels=[3, 3, 3, 3, 3], 
                              conv_strides=[2, 2, 2, 2, (2,1)],
                              latent_space_dim=128)
    
    autoencoder.summary()
    autoencoder.compile(learning_rate)
    history = autoencoder.train(x_train, batch_size, epochs)

    autoencoder._save_history(history, "model")

    return autoencoder

def train_unet(x_train, learning_rate, batch_size, epochs):
    unet = UNET(input_shape=[256,64,1],
                              conv_filters=[64,128,256],
                              conv_kernels=[3], 
                              conv_strides=[2])
    
    unet.summary()
    unet.compile(learning_rate)
    history = unet.train(x_train, batch_size, epochs)

    unet._save_history(history, "model")

    return unet

if __name__ == "__main__":
    x_train = load_fsdd(SPECTROGRAMS_PATH)
    #autoencoder = train_vae(x_train, LEARNING_RATE, BATCH_SIZE, EPOCHS)

    #autoencoder.save("model")

    unet = train_unet(x_train, LEARNING_RATE, BATCH_SIZE, EPOCHS)
    unet.save("model")

    #autoencoder2 = Autoencoder.load("model")
    #autoencoder2.summary()