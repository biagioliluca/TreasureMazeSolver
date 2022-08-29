import emnist
import numpy as np
import csv
from __init__ import table_labels

def get_samples(data, labels, table_labels):
    sub_data = []
    sub_labels = []
    converted_laels = []

    for i in range(len(data)):
        if labels[i] in table_labels.keys():
            sub_data.append(data[i])
            sub_labels.append(labels[i])

    sub_data = np.array(sub_data)
    sub_labels = np.array(sub_labels)

    for i in range(len(sub_labels)):
        sub_labels[i] = table_labels[sub_labels[i]]

    return sub_data, sub_labels

def export_dataset(filename, data, labels):
  csv_file = open(filename, 'w')
  csv_writer = csv.writer(csv_file)

  for entry, label in zip(data, labels):
    x = np.insert(entry, 0, label)
    csv_writer.writerow(x)
  csv_file.close()

def create_dataset(save_dataset_path):
	pre_training_images, pre_training_labels = emnist.extract_training_samples('balanced')
	pre_test_images, pre_test_labels = emnist.extract_test_samples('balanced')

	# extraction of only the interest entries from the dataset adn convert labels in range 0-6
	training_images, training_labels = get_samples(pre_training_images, pre_training_labels, table_labels)
	test_images, test_labels = get_sample(pre_test_images, pre_test_labels, table_labels)

	# pre-processing of the selected dataset
	training_images = training_images.reshape(training_images.shape[0], 784)
	test_images = test_images.reshape(test_images.shape[0], 784)

	export_dataset(save_dataset_path + 'training_dataset.csv', training_images, training_labels)
	export_dataset(save_dataset_path + 'testing_dataset.csv', test_images, test_labels)

def import_dataset(filename):
  csv_file = open(filename, 'r')
  csv_reader = csv.reader(csv_file)

  data = []
  labels = []
  for row in csv_reader:
    entry = [eval(i) for i in row]
    labels.append(entry[0])
    data.append(entry[1:])
  return np.array(data), np.array(labels)

 def dataset_normalization(training_dtset, test_dtset):
 	return training_dtset.astype("float32")/255, test_dtset.astype("float32")/255