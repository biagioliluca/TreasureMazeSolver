# imports
from tensorflow import keras
from keras import models, layers

import numpy as np
import emnist
import csv

from nn_utils import labels_table

def create_model(num_classes):
    '''
      Create the nn-model with default structure
    '''
    model = models.Sequential()

    model.add(layers.Input(shape=784,))
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(num_classes, activation='softmax')) # output layer

    return model


def extract_and_convert_samples(data, labels):
  '''
    Extract the selected data and respective labels using choosen labels
    (labels_table) for our nn-model
  '''
  sub_data = []
  sub_labels = []

  for i in range(len(data)):
      if labels[i] in labels_table.keys():
          sub_data.append(data[i])
          sub_labels.append(labels[i])

  sub_data = np.array(sub_data)
  sub_labels = np.array(sub_labels)

  for i in range(len(sub_labels)):
    sub_labels[i] = labels_table[sub_labels[i]]

  return sub_data, sub_labels
  
def import_dataset(filename):
  '''
    Import dataset from a saved csv file
  '''
  csv_file = open(filename, 'r')
  csv_reader = csv.reader(csv_file)

  data = []
  labels = []
  j = 0
  for row in csv_reader:
    entry = [eval(i) for i in row]
    labels.append(entry[0])
    data.append(entry[1:])
  data = np.array(data).astype('float32')/255
  return data, np.array(labels)

def export_dataset(filename, data, labels):
  '''
    Export dataset on a csv file
  '''
  csv_file = open(filename, 'w')
  csv_writer = csv.writer(csv_file)

  for entry, label in zip(data, labels):
    x = np.insert(entry, 0, label)
    csv_writer.writerow(x)
  csv_file.close()

    

def get_training_dataset():
  '''
  Extract the datasets from EMNIST and select and pre-process them, returning 
  them (samples and labels)
  '''
  pre_training_images, pre_training_labels = emnist.extract_training_samples('balanced')
  # extraction of only the interest entries from the dataset and convert with our labels
  training_images, training_labels = extract_and_convert_samples(pre_training_images, pre_training_labels)
  # pre-processing of the selected dataset
  training_images = training_images.reshape(training_images.shape[0], 784)
  training_images = training_images.astype('float32')/255

  return training_images, training_labels

def get_test_dataset():

  pre_test_images, pre_test_labels = emnist.extract_test_samples('balanced')
  # extraction of only the interest entries from the dataset and convert with our labels
  test_images, test_labels = extract_and_convert_samples(pre_test_images, pre_test_labels)
  # pre-processing of the selected dataset
  test_images = test_images.reshape(test_images.shape[0], 784)
  test_images = test_images.astype('float32')/255

  return test_images, test_labels
