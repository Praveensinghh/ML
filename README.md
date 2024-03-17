# Deep Learning
# Programming Assignment 1
1. Implement the Sparse auto-encoders (AE). Use the MNIST digit dataset for training your network. Per-
form the k-means clustering on the embeddings. To evaluate the performance of the k-means algorithm,
use the available labels in the dataset.

2. Implement variational Auto-encoders. Use the Frey Face dataset to train your network. Sample points
from the learned distribution by varying different latent variables to show that your network has learned
meaningful latent variables. Set the embedding vector size to 20.
Dataset: https://cs.nyu.edu/∼roweis/data/frey rawface.mat

3. Implement GAN and use the Frey Face dataset to train your network. Generate new samples and
comment on the quality of the faces.
Dataset: https://cs.nyu.edu/∼roweis/data/frey rawface.mat

# Dependencies

- Python 3.x
- TensorFlow
- NumPy
- scikit-image
- Matplotlib
- Keras

# Installation

1. Clone this repository:
2. Install the required dependencies:

Download the Frey Raw Face dataset and place it in the project directory.
Download the MNIST dataset and place it in the project directory.

* For 1st question

Install Python and dependencies listed in requirements.txt.
Clone the repository.
Run scripts sequentially: sparse_autoencoder.py, k_means_clustering.py, evaluate_clustering.py.
Customize hyperparameters as needed.

* For 2nd question
* 
Download the Frey Face dataset from the provided link: Frey Face Dataset and place it in the project directory.
Install Python and the dependencies listed in requirements.txt.
Clone the repository.
Run gan_train.py to train the GAN on the Frey Face dataset.
After training, run gan_generate.py to generate new face samples.
Comment on the quality of the generated faces.
Customize hyperparameters as needed.

Files:
gan_train.py: Implementation of the GAN training procedure.
gan_generate.py: Script to generate new face samples using the trained GAN.
frey_face_loader.py: Helper functions for loading and preprocessing the Frey Face dataset.
requirements.txt: Dependencies required to run the code.
README.md: Instructions for usage.

* For 3rd question

Download the Frey Face dataset from the provided link: Frey Face Dataset and place it in the project directory.
Install Python and the dependencies listed in requirements.txt.
Clone the repository.
Run gan_train.py to train the GAN on the Frey Face dataset.
Once training is complete, execute gan_generate.py to generate new face samples.
Assess the quality of the generated faces.
Customize hyperparameters to improve results if necessary.

Files:
gan_train.py: GAN training implementation.
gan_generate.py: Script for generating new face samples.
frey_face_loader.py: Functions for loading and preprocessing the Frey Face dataset.
requirements.txt: Dependencies required to run the code.
README.md: Instructions for usage and overview of the repository.
