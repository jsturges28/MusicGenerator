import tensorflow as tf
from tensorflow import keras
from keras import Model
from keras.layers import Input, Conv2D, ReLU, BatchNormalization, Flatten, Dense, \
    Reshape, Conv2DTranspose, Activation, Lambda, MaxPooling2D, UpSampling2D, Concatenate, \
    SeparableConv2D
from keras import backend as K
from keras.optimizers import Adam
from keras.losses import MeanSquaredError, KLDivergence
import numpy as np
import os
import pickle
import librosa

tf.compat.v1.disable_eager_execution() # Prevent evaluating operations before graph is completely built

class VUNET:
    """
    VUNET architecture
    """

    def __init__(self, input_shape, conv_filters, conv_kernels, conv_strides, latent_space_dim):
        self.input_shape = input_shape # example: [28, 28, 1]
        self.conv_filters = conv_filters # [2,4,8]
        self.conv_kernels = conv_kernels # [3, 5, 3] 3x3, 5x5, 3x3
        self.conv_strides = conv_strides # [1, 2, 2]
        self.latent_space_dim = latent_space_dim # 2
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
        parameters_path = os.path.join(save_folder, "parameters_vunet.pkl")
        with open(parameters_path, "rb") as f:
            parameters = pickle.load(f)

        vunet = VUNET(*parameters)
        weights_path = os.path.join(save_folder, "weights_vunet.h5")
        vunet.load_weights(weights_path)

        return vunet
    
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
            self.latent_space_dim 
        ]
        save_path = os.path.join(save_folder, "parameters_vunet.pkl")
        with open(save_path, "wb") as f:
            pickle.dump(parameters, f)

    def _save_weights(self, save_folder):
        save_path = os.path.join(save_folder, "weights_vunet.h5")
        self.model.save_weights(save_path)

    def _save_history(self, history, save_folder="."):
        history_dict = history.history  # Convert history object to dict

        # Create save path
        save_path = os.path.join(save_folder, "history_vunet.pkl")

        # Save history data
        with open(save_path, "wb") as f:
            pickle.dump(history_dict, f)

    def _add_bottleneck(self, x, layer_index):
        """Flatten data and add bottleneck with Gaussian sampling (Dense layer)."""
        self._shape_before_bottleneck = K.int_shape(x)[1:] # [2, 7, 7, 32] -> [7, 7, 32] ignore the first dim (batch size 2)
        x = Flatten()(x)
        self.mu = Dense(self.latent_space_dim, name="mu_" + str(layer_index))(x)
        self.log_variance = Dense(self.latent_space_dim, name="log_variance_" + str(layer_index))(x)

        def sample_point_from_normal_distribution(args):
            mu, log_variance = args
            epsilon = K.random_normal(shape=K.shape(self.mu), mean=0.0, stddev=1.0)
            sampled_point = mu + K.exp(log_variance / 2) * epsilon
            return sampled_point


        x = Lambda(sample_point_from_normal_distribution, name="encoder_output_" + str(layer_index))([self.mu, self.log_variance])
        x = Dense(np.prod(self._shape_before_bottleneck))(x)
        x = Reshape(self._shape_before_bottleneck)(x)

        return x

    def _build(self):
        self._build_vunet()

    def _build_vunet(self):
        tensor_stack = []
        self.encoder_shapes = []
        input_tensor = Input(shape=self.input_shape, name="input")
        tensor = input_tensor

        for i in range(len(self.conv_filters) - 1):
            tensor = Conv2D(self.conv_filters[0], kernel_size=self.conv_kernels[0], padding='same', activation='relu')(tensor)
            tensor = Conv2D(self.conv_filters[0], kernel_size=self.conv_kernels[0], padding='same', activation='relu')(tensor)
            self.encoder_shapes.append(K.int_shape(tensor)[1:])
            tensor = self._add_bottleneck(tensor, i)
            tensor_stack.append(tensor)
            tensor = MaxPooling2D(pool_size=2, strides=2)(tensor)

        #32x32
        tensor = Conv2D(self.conv_filters[2], kernel_size=self.conv_kernels[0], padding='same', activation='relu')(tensor)
        tensor = Conv2D(self.conv_filters[2], kernel_size=self.conv_kernels[0], padding='same', activation='relu')(tensor)

        for i in reversed(range(len(self.encoder_shapes))):
            tensor = UpSampling2D(size=2)(tensor)
            # Get the corresponding encoder output shape
            encoder_shape = self.encoder_shapes[i]

            latent_vector = tensor_stack.pop()
            # Reshape your latent vector to match the encoder output shape
            latent_vector = Reshape(encoder_shape)(latent_vector)

            # Concatenate latent vector with the decoder output
            tensor = Concatenate()([tensor, latent_vector])
            tensor = Conv2D(self.conv_filters[i], kernel_size=self.conv_kernels[0], padding='same', activation='relu')(tensor)
            tensor = Conv2D(self.conv_filters[i], kernel_size=self.conv_kernels[0], padding='same', activation='relu')(tensor)
        
        # output
        tensor = Conv2D(1, kernel_size=self.conv_kernels[0], padding='same', activation='sigmoid')(tensor)

        self.model = Model(inputs=input_tensor, outputs=tensor)
    

if __name__ == '__main__':
    vunet = VUNET(input_shape=[64,2584,1],
                              conv_filters=[64, 128, 256],
                              conv_kernels=[3], 
                              conv_strides=[1, 2, 2, 1],
                              latent_space_dim=2)
    vunet.summary()