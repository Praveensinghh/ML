#!/usr/bin/env python
# coding: utf-8

# In[3]:


import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras.utils import plot_model
import matplotlib.pyplot as plt
import numpy as np
import time
import requests
import glob
import scipy.io
import numpy as np
import os



url_frey = "https://cs.nyu.edu/~roweis/data/frey_rawface.mat"
route = "frey_rawface.mat"

def dataset(url_frey, route):
    try:
        response = requests.get(url_frey)
        if response.status_code == 200:
            with open(route, 'wb') as file:
                file.write(response.content)
        else:
            print(f"failed")
    except Exception as e:
        print(f"error : {e}")
dataset(url_frey, route)

mat = scipy.io.loadmat('frey_rawface.mat', squeeze_me=True, struct_as_record=False)
ff = mat["ff"].T.reshape((-1, 28, 20, 1))
np.random.seed(42)

trainx = ff.astype('float32')
dataset = (trainx - 127.5) / 127.5


def DESCR(in_shape=(28,20,1)):
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(128, (3,3), padding="same", input_shape=in_shape))
    model.add(layers.LeakyReLU(alpha=0.2))

    model.add(layers.Conv2D(256, (3,3), strides=(2,2), padding="same"))
    model.add(layers.LeakyReLU(alpha=0.2))

    model.add(layers.Conv2D(512, (3,3), strides=(2,2), padding="same"))
    model.add(layers.LeakyReLU(alpha=0.2))

    model.add(layers.Flatten())
    model.add(layers.Dropout(0.4))
    model.add(layers.Dense(1, activation="sigmoid"))

    opt = optimizers.Adam(lr=0.002, beta_1=0.5)
    model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])
    return model

def GENERATOR(latent_dim):
    model = tf.keras.Sequential()
    model.add(layers.Dense(256*7*5, input_dim=latent_dim))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Reshape((7,5,256)))
    model.add(layers.Conv2DTranspose(128, (3,3), strides=(2,2), padding="same"))
    model.add(layers.LeakyReLU(alpha=0.2))

    model.add(layers.Conv2DTranspose(128, (3,3), strides=(2,2), padding="same"))
    model.add(layers.LeakyReLU(alpha=0.2))

    model.add(layers.Conv2D(1, (3,3), activation="tanh", padding="same"))
    return model


Model1 = DESCR()
print(Model1.summary())

latent_dim = 165
model = GENERATOR(latent_dim)
print(model.summary())

def gen_latent(latent_dim, n_samples):
    x_input = np.random.randn(latent_dim * n_samples)
    x_input = x_input.reshape(n_samples, latent_dim)
    return x_input

def fake_data_gen(g_model, latent_dim, n_samples):
    x_input = gen_latent(latent_dim, n_samples)
    X = g_model.predict(x_input)
    y= np.zeros((n_samples, 1))
    return X,y

def generate_real_data(dataset, n_samples):
    ix = np.random.randint(0, dataset.shape[0], n_samples)
    X = dataset[ix]
    y = np.ones((n_samples, 1))
    return X,y


def ganfunc(g_model, d_model):
    d_model.trainable = False
    model = tf.keras.Sequential()
    model.add(g_model)
    model.add(d_model)
    opt = optimizers.Adam(lr=0.002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt)
    return model

latent_dim = 165
d_model = DESCR()
g_model = GENERATOR(latent_dim)
gan_model = ganfunc(g_model, d_model)
print(gan_model.summary())

def save_images(examples, epoch, n=4):
    examples = (examples + 1) / 2.0
    for i in range(n * n):
        plt.subplot(n, n, 1+i)
        plt.axis('off')
        plt.imshow(examples[i])
        file = f'generated_plot_e{epoch+1}'
    plt.savefig(file)
    plt.close()

def acc(epoch, g_model, d_model, dataset, latent_dim, n_samples=150):
    X_real, y_real = generate_real_data(dataset, n_samples)
    _, acc_real = d_model.evaluate(X_real, y_real, verbose=0)
    X_fake, y_fake = fake_data_gen(g_model, latent_dim, n_samples)
    _, acc_fake = d_model.evaluate(X_fake, y_fake, verbose=0)
    print(f'Accuracy- real: {acc_real*100:.0f}%, fake: {acc_fake*100:.0f}%')
    save_images(X_fake, epoch)
    filename = f'generator_model_{epoch+1}.h5'
    g_model.save(filename)


def model(g_model, d_model, gan_model, dataset, latent_dim, n_epochs=50, n_batch=256):
    bat_per_epo = int(dataset.shape[0] / n_batch)
    half_batch = int(n_batch/2)
    for i in range(n_epochs):
        for j in range(bat_per_epo):
            X_real, y_real = generate_real_data(dataset, half_batch)
            d_loss1, _ = d_model.train_on_batch(X_real, y_real)
            X_fake, y_fake = fake_data_gen(g_model, latent_dim, half_batch)
            d_loss2, _ = d_model.train_on_batch(X_fake, y_fake)
            X_gan = gen_latent(latent_dim, n_batch)
            y_gan = np.ones((n_batch,1))
            g_loss = gan_model.train_on_batch(X_gan, y_gan)
        print(f'>{i+1}, {(j+1) / (bat_per_epo)}, d1={d_loss1:.3f}, d2={d_loss2:.3f} g={g_loss:.3f}')
    acc(i, g_model, d_model, dataset, latent_dim)


latent_dim = 165
d_model = DESCR()
g_model = GENERATOR(latent_dim)
gan_model = ganfunc(g_model, d_model)
model(g_model, d_model, gan_model, dataset, latent_dim)





