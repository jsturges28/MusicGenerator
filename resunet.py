import tensorflow as tf
from tensorflow import keras
from keras import Model
from keras.layers import Input, Conv2D, ReLU, BatchNormalization, Flatten, Dense, \
    Reshape, Conv2DTranspose, Activation, Lambda, MaxPooling2D, UpSampling2D, Concatenate
from keras import backend as K
from keras.optimizers import Adam
from keras.losses import MeanSquaredError, KLDivergence
import numpy as np
import os
import pickle
import librosa

tf.compat.v1.disable_eager_execution() # Prevent evaluating operations before graph is completely built

class RESUNET:
    """
    RESUNET architecture
    """

    def __init__(self, input_shape, conv_filters, conv_kernels, conv_strides):
        self.input_shape = input_shape # example: [28, 28, 1]
        self.conv_filters = conv_filters # [2,4,8]
        self.conv_kernels = conv_kernels # [3, 5, 3] 3x3, 5x5, 3x3
        self.conv_strides = conv_strides # [1, 2, 2]
        self.reconstruction_loss_weight = 1000000 # alpha value
        #self.callback = tf.keras.callbacks.EarlyStopping(monitor='loss', min_delta=0.01, patience=2)

        self.encoder = None
        self.decoder = None
        self.model = None

        self._num_conv_layers = len(conv_filters)
        self._shape_before_bottleneck = None
        self._model_input = None
        
        self._build()

    def summary(self):
        self.model.summary()

    def compile(self, learning_rate=0.0001):
        optimizer = Adam(learning_rate=learning_rate)
        self.model.compile(optimizer=optimizer, loss=self._calculate_spectral_log_loss, metrics=['MeanSquaredError', 'MeanAbsoluteError'])

    def train(self, x_train, batch_size, num_epochs):
        history = self.model.fit(x_train, x_train, batch_size=batch_size, epochs=num_epochs, shuffle=True)
        return history

    def save(self, save_folder="."):
        self._create_folder_if_it_doesnt_exist(save_folder)
        self._save_parameters(save_folder)
        self._save_weights(save_folder)

    def load_weights(self, weights_path):
        self.model.load_weights(weights_path)

    def reconstruct(self, images):
        reconstructed_images = self.model.predict(images)
        return reconstructed_images

    @classmethod
    def load(cls, save_folder="."):
        parameters_path = os.path.join(save_folder, "parameters_resunet.pkl")
        with open(parameters_path, "rb") as f:
            parameters = pickle.load(f)

        resunet = RESUNET(*parameters)
        weights_path = os.path.join(save_folder, "weights_resunet.h5")
        resunet.load_weights(weights_path)

        return resunet
    
    def _calculate_spectral_log_loss(self, y_true, y_pred, eps=1e-6, norm='l1'):
        error = y_true - y_pred
        # Compute the L1 or L2 loss
        if norm == 'l1':
            return K.mean(K.abs(error), axis=[1,2,3])
        elif norm == 'l2':
            return K.mean(K.square(error), axis=[1,2,3])
        else:
            raise ValueError("Invalid norm type: must be either 'l1' or 'l2'")
    
    def _create_folder_if_it_doesnt_exist(self, folder):
        if not os.path.exists(folder):
            os.makedirs(folder)

    def _save_parameters(self, save_folder):
        parameters = [
            self.input_shape,
            self.conv_filters,
            self.conv_kernels,
            self.conv_strides,
        ]
        save_path = os.path.join(save_folder, "parameters_resunet.pkl")
        with open(save_path, "wb") as f:
            pickle.dump(parameters, f)

    def _save_weights(self, save_folder):
        save_path = os.path.join(save_folder, "weights_resunet.h5")
        self.model.save_weights(save_path)

    def _save_history(self, history, save_folder="."):
        history_dict = history.history  # Convert history object to dict

        # Create save path
        save_path = os.path.join(save_folder, "history_resunet.pkl")

        # Save history data
        with open(save_path, "wb") as f:
            pickle.dump(history_dict, f)

    def bn_act(self, x, act=True):
        x = keras.layers.BatchNormalization()(x)
        if act == True:
            x = keras.layers.Activation("relu")(x)
        return x

    def conv_block(self, x, filters, kernel_size=(3, 3), padding="same", strides=1):
        conv = self.bn_act(x)
        conv = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides)(conv)
        return conv

    def stem(self, x, filters, kernel_size=(3, 3), padding="same", strides=1):
        conv = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides)(x)
        conv = self.conv_block(conv, filters, kernel_size=kernel_size, padding=padding, strides=strides)
        
        shortcut = keras.layers.Conv2D(filters, kernel_size=(1, 1), padding=padding, strides=strides)(x)
        shortcut = self.bn_act(shortcut, act=False)
        
        output = keras.layers.Add()([conv, shortcut])
        return output

    def residual_block(self, x, filters, kernel_size=(3, 3), padding="same", strides=1):
        res = self.conv_block(x, filters, kernel_size=kernel_size, padding=padding, strides=strides)
        res = self.conv_block(res, filters, kernel_size=kernel_size, padding=padding, strides=1)
        
        shortcut = keras.layers.Conv2D(filters, kernel_size=(1, 1), padding=padding, strides=strides)(x)
        shortcut = self.bn_act(shortcut, act=False)
        
        output = keras.layers.Add()([shortcut, res])
        return output

    def upsample_concat_block(self, x, xskip):
        u = keras.layers.UpSampling2D((2, 2))(x)
        c = keras.layers.Concatenate()([u, xskip])
        return c

    def _build(self):
        self._build_resunet()

    def _build_resunet(self):
        f = self.conv_filters
        inputs = keras.layers.Input(shape=self.input_shape, name="input")
        
        ## Encoder
        e0 = inputs
        e1 = self.stem(e0, f[0])
        e2 = self.residual_block(e1, f[1], strides=2)
        e3 = self.residual_block(e2, f[2], strides=2)
        e4 = self.residual_block(e3, f[3], strides=2)
        e5 = self.residual_block(e4, f[4], strides=2)
        
        ## Bridge
        b0 = self.conv_block(e5, f[4], strides=1)
        b1 = self.conv_block(b0, f[4], strides=1)
        
        ## Decoder
        u1 = self.upsample_concat_block(b1, e4)
        d1 = self.residual_block(u1, f[4])
        
        u2 = self.upsample_concat_block(d1, e3)
        d2 = self.residual_block(u2, f[3])
        
        u3 = self.upsample_concat_block(d2, e2)
        d3 = self.residual_block(u3, f[2])
        
        u4 = self.upsample_concat_block(d3, e1)
        d4 = self.residual_block(u4, f[1])
        
        outputs = keras.layers.Conv2D(1, kernel_size=self.conv_kernels[0], padding="same", activation="sigmoid")(d4)
        model = keras.models.Model(inputs, outputs)
        return model
    

if __name__ == '__main__':
    resunet = RESUNET(input_shape=[64,2584,1],
                              conv_filters=[16, 32, 64, 128, 256],
                              conv_kernels=[3], 
                              conv_strides=[1, 2, 2, 1])
    resunet.summary()