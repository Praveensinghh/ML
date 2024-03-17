# ML
Progressive Growing GAN (PGGAN) for Frey Raw Face Dataset

Author: Praveen

---

Overview

This repository contains code for training a Progressive Growing Generative Adversarial Network (PGGAN) on the Frey Raw Face dataset. The PGGAN is an extension of the traditional GAN architecture that gradually increases the resolution of generated images during training, leading to more realistic and high-resolution outputs.

Dependencies

- Python 3.x
- TensorFlow
- NumPy
- scikit-image
- Matplotlib
- Keras

Installation

1. Clone this repository:

Dataset

Download the Frey Raw Face dataset and place it in the project directory.

Training

To train the PGGAN on the Frey Raw Face dataset, follow these steps:

1. Navigate to the project directory.
2. Run the training script.
3. Monitor the training progress and generated images.

Code Structure

- train_pggan.py: Main script for training the PGGAN.
- utils.py: Contains utility functions for data loading, model definition, and training.

References

- Progressive Growing of GANs for Improved Quality, Stability, and Variation: Tero Karras et al., 2018.
- Frey Raw Face Dataset: Geoffrey E. Hinton's website.

License

This project is licensed under the MIT License - see the LICENSE file for details.

Contact

For any inquiries or issues, please contact Praveen at praveen@email.com.
