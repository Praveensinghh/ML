import os
import time
import tensorflow as tf
from tensorflow.keras import layers
from IPython import display
import matplotlib.pyplot as plt
import numpy as np
from keras import backend as K
from tensorflow import keras
import scipy.io

img_rows, img_cols = 28, 20
mat = scipy.io.loadmat('frey_rawface.mat', squeeze_me=True, struct_as_record=False)
ff = mat["ff"].T.reshape((-1, 560))
ff = ff / 255.
x_train = ff[:1800]
x_test = ff[1800:]
tf.random.set_seed(42)

def sampling(args):
    z_mean, z_log_var = args
    batch = tf.shape(z_mean)[0]
    dim = tf.shape(z_mean)[1]
    epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
    return z_mean + tf.exp(0.5 * z_log_var) * epsilon

latent_dim = 20
encoder_inputs = keras.Input(shape=(560,))
x = layers.Dense(256, activation='relu')(encoder_inputs)
z_mean = layers.Dense(latent_dim, name="z_mean")(x)
z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
# z = sampling()([z_mean, z_log_var])
z = layers.Lambda(sampling)([z_mean, z_log_var])
encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")

latent_inputs = keras.Input(shape=(latent_dim,))
x = layers.Dense(256, activation='relu')(latent_inputs)
decoder_outputs = layers.Dense(560, activation='sigmoid')(x)
decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")

encoder_outputs = encoder(encoder_inputs)
outputs = decoder(encoder_outputs[2])
vae = keras.Model(encoder_inputs, outputs, name="vae")

# Reconstruction loss
reconstruction_loss = tf.losses.mean_squared_error(encoder_inputs, outputs)
reconstruction_loss = reconstruction_loss * 560

# KL Divergence loss
kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
kl_loss = -0.5 * tf.reduce_sum(kl_loss, axis=-1)
vae_loss = tf.reduce_mean(reconstruction_loss + kl_loss)
optimizer = keras.optimizers.Adam(learning_rate=0.001)
vae.add_loss(vae_loss)
vae.compile(optimizer=optimizer)
vae.fit(x_train, x_train, epochs=12, batch_size=50, validation_data=(x_test, x_test))
vae.save('model.h5')
