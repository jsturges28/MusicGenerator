import tensorflow as tf
from tensorflow import keras
from keras import Model
from keras.layers import Input, Conv2D, ReLU, BatchNormalization, Flatten, Dense, \
    Reshape, Conv2DTranspose, Activation, Lambda, MaxPooling2D, UpSampling2D, Concatenate, \
    LeakyReLU, Dropout, GlobalAveragePooling2D
from keras import backend as K
from keras.optimizers import Adam
from keras.losses import MeanSquaredError, KLDivergence, BinaryCrossentropy
import numpy as np
import os
import pickle
import librosa

tf.compat.v1.disable_eager_execution() # Prevent evaluating operations before graph is completely built

class GANUNET:
    """
    GANUNET architecture
    """

    def __init__(self, input_shape, conv_filters, conv_kernels, conv_strides):
        self.input_shape = input_shape # example: [28, 28, 1]
        self.conv_filters = conv_filters # [2,4,8]
        self.conv_kernels = conv_kernels # [3, 5, 3] 3x3, 5x5, 3x3
        self.conv_strides = conv_strides # [1, 2, 2]
        self.reconstruction_loss_weight = 1000000 # alpha value
        #self.callback = tf.keras.callbacks.EarlyStopping(monitor='loss', min_delta=0.01, patience=2)
        self.bce = keras.losses.BinaryCrossentropy()
        self.batch_size = 16

        self.generator = None
        self.discriminator = None
        self.model = None

        self._num_conv_layers = len(conv_filters)
        self._shape_before_bottleneck = None
        self._model_input = None
        
        self._build()

    def summary(self):
        self.generator.summary()
        self.discriminator.summary()
        self.model.summary()

    def compile(self, learning_rate=0.0001):
        optimizer = Adam(learning_rate=learning_rate)

        # Compile the discriminator with its own loss and accuracy metric
        self.discriminator.compile(optimizer=optimizer, loss=self._calculate_discriminator_loss, metrics=['accuracy'])

        # For the combined model, we will only train the generator
        self.discriminator.trainable = False

        # The combined model (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.model.compile(loss=self._calculate_generator_loss, optimizer=optimizer)

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
        parameters_path = os.path.join(save_folder, "parameters_ganunet.pkl")
        with open(parameters_path, "rb") as f:
            parameters = pickle.load(f)

        ganunet = GANUNET(*parameters)
        weights_path = os.path.join(save_folder, "weights_ganunet.h5")
        ganunet.load_weights(weights_path)

        return ganunet
    
    def _calculate_discriminator_loss(self, y_true, y_pred):
        return self.bce(y_true, y_pred)
    
    def _calculate_generator_loss(self, y_true, y_pred, discriminator_output):
        # The generator tries to make the discriminator output 1 for its generated images
        valid = np.ones((self.input_shape))
        gen_loss = tf.convert_to_tensor(self.bce(valid, discriminator_output))

        # The generator also tries to minimize the spectral log loss
        spec_loss = self._calculate_spectral_log_loss(y_true, y_pred)

        lambda_gen = 1.0  # weight for the generator's binary cross-entropy loss
        lambda_spec = 1.0  # weight for the spectral log loss

        return lambda_gen * gen_loss + lambda_spec * spec_loss
    
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
        save_path = os.path.join(save_folder, "parameters_ganunet.pkl")
        with open(save_path, "wb") as f:
            pickle.dump(parameters, f)

    def _save_weights(self, save_folder):
        save_path = os.path.join(save_folder, "weights_ganunet.h5")
        self.model.save_weights(save_path)

    def _save_history(self, history, save_folder="."):
        history_dict = history.history  # Convert history object to dict

        # Create save path
        save_path = os.path.join(save_folder, "history_ganunet.pkl")

        # Save history data
        with open(save_path, "wb") as f:
            pickle.dump(history_dict, f)

    def _build(self):
        self._build_generator()
        self._build_discriminator()
        self._build_ganunet()

    def _build_generator(self):
        tensor_stack = []
        generator_input = Input(shape=self.input_shape, name="input")
        tensor = generator_input
        self._model_input = generator_input

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

        #print("t2 output: ", tensor.shape)

        #64x64
        tensor = Conv2D(self.conv_filters[1], kernel_size=self.conv_kernels[0], padding='same', activation='relu')(tensor)
        tensor = Conv2D(self.conv_filters[1], kernel_size=self.conv_kernels[0], padding='same', activation='relu')(tensor)
        tensor = UpSampling2D(size=2)(tensor)

        #128x128
        
        # skip connection
        t3 = tensor_stack.pop()

        tensor = Concatenate()([tensor, t3])

        #print("t3 output: ", tensor.shape)

        #128x128
        tensor = Conv2D(self.conv_filters[0], kernel_size=self.conv_kernels[0], padding='same', activation='relu')(tensor)
        tensor = Conv2D(self.conv_filters[0], kernel_size=self.conv_kernels[0], padding='same', activation='relu')(tensor)
        
        # output
        tensor = Conv2D(1, kernel_size=self.conv_kernels[0], padding='same', activation='sigmoid')(tensor)

        self.generator_output_shape = tensor.shape

        #print("Generator output: ", tensor.shape)

        self.generator = Model(inputs=generator_input, outputs=tensor, name="generator")

    def _build_discriminator(self):
        discriminator_input = self._add_discriminator_input()
        discriminator_layers = self._add_discriminator_layers(discriminator_input)
        discriminator_output = self._add_discriminator_output(discriminator_layers)

        self.discriminator = Model(discriminator_input, discriminator_output, name="discriminator")

    def _add_discriminator_input(self):
        x = Input(shape=self.generator_output_shape[1:], name="discriminator_input")
        #print("Shape of disc. input: ", x.shape)
        return x
    
    def _add_discriminator_layers(self, discriminator_input):
        
        x = Conv2D(64, (5, 5), strides=(2, 2), padding='same')(discriminator_input)
        x = LeakyReLU()(x)
        x = Dropout(0.3)(x)

        x = Conv2D(128, (5, 5), strides=(2, 2), padding='same')(x)
        x = LeakyReLU()(x)
        x = Dropout(0.3)(x)

        x = GlobalAveragePooling2D()(x)
        x = Dense(1)(x)

        return x
    
    def _add_discriminator_output(self, layers):
        output_layer = Activation("sigmoid", name="sigmoid_layer")(layers)

        return output_layer
    
    def _build_ganunet(self):
        model_input = self._model_input
        model_output = self.discriminator(self.generator(model_input))

        self.model = Model(model_input, model_output, name="ganunet")




    

if __name__ == '__main__':
    ganunet = GANUNET(input_shape=[64,2584,1],
                              conv_filters=[64,128,256],
                              conv_kernels=[3], 
                              conv_strides=[1, 2, 2, 1])
    ganunet.summary()