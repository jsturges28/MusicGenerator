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

class UNET:
    """
    UNET architecture
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
        parameters_path = os.path.join(save_folder, "parameters_unet.pkl")
        with open(parameters_path, "rb") as f:
            parameters = pickle.load(f)

        unet = UNET(*parameters)
        weights_path = os.path.join(save_folder, "weights_unet.h5")
        unet.load_weights(weights_path)

        return unet
    
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
        save_path = os.path.join(save_folder, "parameters_unet.pkl")
        with open(save_path, "wb") as f:
            pickle.dump(parameters, f)

    def _save_weights(self, save_folder):
        save_path = os.path.join(save_folder, "weights_unet.h5")
        self.model.save_weights(save_path)

    def _save_history(self, history, save_folder="."):
        history_dict = history.history  # Convert history object to dict

        # Create save path
        save_path = os.path.join(save_folder, "history_unet.pkl")

        # Save history data
        with open(save_path, "wb") as f:
            pickle.dump(history_dict, f)

    def _build(self):
        self._build_unet()

    def _build_unet(self):
        tensor_stack = []
        input_tensor = Input(shape=self.input_shape, name="input")
        tensor = input_tensor

        #128x128
        tensor = Conv2D(self.conv_filters[0], kernel_size=self.conv_kernels[0], padding='same', activation='relu')(tensor)
        tensor = Conv2D(self.conv_filters[0], kernel_size=self.conv_kernels[0], padding='same', activation='relu')(tensor)
        tensor_stack.append(tensor)
        tensor = MaxPooling2D(pool_size=2, strides=2)(tensor)

        #64x64
        tensor = Conv2D(self.conv_filters[1], kernel_size=self.conv_kernels[0], padding='same', activation='relu')(tensor)
        tensor = Conv2D(self.conv_filters[1], kernel_size=self.conv_kernels[0], padding='same', activation='relu')(tensor)
        tensor_stack.append(tensor) # t2
        tensor = MaxPooling2D(pool_size=2, strides=2)(tensor)

        #32x32
        tensor = Conv2D(self.conv_filters[2], kernel_size=self.conv_kernels[0], padding='same', activation='relu')(tensor)
        tensor = Conv2D(self.conv_filters[2], kernel_size=self.conv_kernels[0], padding='same', activation='relu')(tensor)
    
        tensor = UpSampling2D(size=2)(tensor)

        #64x64

        #skip connections
        t2 = tensor_stack.pop() # removes last item from tensor stack and returns it
        tensor = Concatenate()([tensor, t2]) # Concat along channels

        #64x64
        tensor = Conv2D(self.conv_filters[1], kernel_size=self.conv_kernels[0], padding='same', activation='relu')(tensor)
        tensor = Conv2D(self.conv_filters[1], kernel_size=self.conv_kernels[0], padding='same', activation='relu')(tensor)
        tensor = UpSampling2D(size=2)(tensor)

        #128x128
        
        # skip connection
        t3 = tensor_stack.pop()

        tensor = Concatenate()([tensor, t3])

        #128x128
        tensor = Conv2D(self.conv_filters[0], kernel_size=self.conv_kernels[0], padding='same', activation='relu')(tensor)
        tensor = Conv2D(self.conv_filters[0], kernel_size=self.conv_kernels[0], padding='same', activation='relu')(tensor)
        
        # output
        tensor = Conv2D(1, kernel_size=self.conv_kernels[0], padding='same', activation='sigmoid')(tensor)

        self.model = Model(inputs=input_tensor, outputs=tensor)
    

if __name__ == '__main__':
    unet = UNET(input_shape=[32,32,1],
                              conv_filters=[64,128,256],
                              conv_kernels=[3], 
                              conv_strides=[1, 2, 2, 1])
    unet.summary()