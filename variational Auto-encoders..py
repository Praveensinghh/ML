import numpy as np  # Importing the NumPy library for numerical computations
import matplotlib.pyplot as plt  # Importing the Matplotlib library for creating visualisations
from keras.layers import Input, Dense, Lambda  # Importing specific layers from Keras
from keras.models import Model  # Importing the Model class from Keras for defining neural network models
from tensorflow.keras.losses import MeanSquaredError  # Importing the Mean Squared Error loss function from TensorFlow's Keras implementation
from keras import backend as K  # Importing the Keras backend module for accessing backend operations
from keras.datasets import frey_faces  # Importing the Frey Face dataset for testing machine learning algorithms

# Load the Frey Face dataset
(x_train, _), (x_test, _) = frey_faces.load_data()  # Loading the training and testing data

# Normalize pixel values to [0, 1]
x_train = x_train.astype('float32') / 255.  # Normalising the training data
x_test = x_test.astype('float32') / 255.  # Normalising the testing data

# Flatten the images
input_dim = np.prod(x_train.shape[1:])  # Calculating input dimensionality as the product of image dimensions
x_train_flat = x_train.reshape((len(x_train), input_dim))  # Flattening the training images
x_test_flat = x_test.reshape((len(x_test), input_dim))  # Flattening the testing images

# Set the size of the latent space
latent_dim = 20  # Defining the dimensionality of the latent space

# Encoder network
input_layer = Input(shape=(input_dim,))  # Defining the input layer with the specified shape
encoded_layer = Dense(128, activation='relu')(input_layer)  # Defining the encoded layer with ReLU activation
z_mean_layer = Dense(latent_dim)(encoded_layer)  # Defining the layer for the mean of the latent space
z_log_var_layer = Dense(latent_dim)(encoded_layer)  # Defining the layer for the log variance of the latent space

# Sampling function
def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean=0., stddev=1.)
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

z_layer = Lambda(sampling)([z_mean_layer, z_log_var_layer])  # Defining the sampling layer using the Lambda layer

# Decoder network
decoded_layer = Dense(128, activation='relu')(z_layer)  # Defining the decoded layer with ReLU activation
output_layer = Dense(input_dim, activation='sigmoid')(decoded_layer)  # Defining the output layer with sigmoid activation

# VAE model
vae = Model(input_layer, output_layer)  # Defining the VAE model using input and output layers

# Define the loss function
reconstruction_loss = MeanSquaredError()(input_layer, output_layer)  # Calculating the reconstruction loss
kl_loss = -0.5 * K.sum(1 + z_log_var_layer - K.square(z_mean_layer) - K.exp(z_log_var_layer), axis=-1)  # Calculating the KL divergence loss
vae_loss = K.mean(reconstruction_loss + kl_loss)  # Calculating the total VAE loss
vae.add_loss(vae_loss)  # Adding VAE loss to the model

# Compile the model
vae.compile(optimizer='adam')  # Compiling the VAE model using Adam optimizer

# Train the VAE
vae.fit(x_train_flat, epochs=50, batch_size=128, validation_data=(x_test_flat, None))  # Training the VAE model

# Generate samples from the learned distribution
n_samples = 10  # Defining the number of samples to generate
random_latent_vectors = np.random.normal(size=(n_samples, latent_dim))  # Generating random latent vectors
generated_images = vae.predict(random_latent_vectors)  # Generating images from random latent vectors

# Display the generated samples
plt.figure(figsize=(10, 2))  # Setting the figure size for displaying generated samples
for i in range(n_samples):
    ax = plt.subplot(1, n_samples, i + 1)  # Creating a subplot for each generated sample
    plt.imshow(generated_images[i].reshape(28, 20), cmap='gray')  # Displaying the generated sample as a grayscale image
    plt.axis('off')  # Turning off axis labels
plt.show()  # Displaying the generated samples
