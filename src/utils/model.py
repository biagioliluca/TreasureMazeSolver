'''
    Questo file.py contiene tutte le classi e le funzioni necessarie per la
    creazione, addestramento e salvataggio della rete neurale.
'''
# imports
from tensorflow import keras
from keras import models, layers

import emnist

import numpy as np

# funzione per creare il modello
def create_model(num_classes):
    '''
        Funzione che istanzia un modello di rete neurale attraverso Keras.
    '''
    model = models.Sequential()

    model.add(layers.Input(shape=784,))
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(num_classes, activation='softmax')) # output layer

    return model

# funzione per addestrare il modello e salvarlo
def extract_samples(dataset, labels, selected_classes):
    sub_dataset = []
    sub_labels = []

    for i in range(len(dataset)):
        if labels[i] in selected_classes:
            sub_dataset.append(dataset[i])
            sub_labels.append(labels[i])

    return np.array()


# funzione per salvarlo (necessario?)


# funzione per addestrare il modello


#