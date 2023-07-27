# 3DGCN_augmentation: Three-Dimensionally Embedded Graph Convolutional Network (3DGCN) for Molecule Interpretation with rotation variance-based data augmentation

This is an implementation of our paper "Rotational Variance-Based Data Augmentation in 3D Graph Convolutional Network":

Jihoo Kim, Yeji Kim, Eok Kyun Lee, Chong Hak Chae, Kwangho Lee, Won June Kim, Insung S. Choi, [Rotational Variance-Based Data Augmentation in 3D Graph Convolutional Network] (Chem. Asian J. 2021, 16(18), 2610-2613.)

## Requirements

* Python 3.6.1
* Tensorflow 1.15
* Keras 2.25
* RDKit
* scikit-learn

## Data

* BACE dataset

## Models

The `models` folder contains python scripts for building, training, and evaluation of the 3DGCN model with rotation variance-data augmentation.

The 'dataset.py' cleans and prepares the dataset for the model training with data agumentation.
The 'layer.py' and 'model.py' build the model structure.
The 'loss.py' and 'callbacks.py' assign the loss and metrics that we wanted to use.
The 'trainer.py' and 'bace.py' are for training of the model.
The 'rotational_invariance.py' evaluates the trained model with ligand rotations.
