from nn_utils import model as md
from nn_utils import NUM_CLASSES

import argparse
from pathlib import Path
import os

from tensorflow.keras.optimizers import SGD

MODELS_PATH = os.path.join(Path(__file__).resolve().parent.parent.parent, 'models')

EPOCHS = 16
BATCH_SIZE = 64
LEARNING_R = 0.1
DECAY_R = LEARNING_R / EPOCHS
MOMENTUM = 0.8



# define argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--training_dataset_path', type=Path, help='enter training dataset path, or leave it blank to extract it from EMNIST')
parser.add_argument('--test_dataset_path', type=Path, help='enter test dataset path, or leave it blank to extract it from EMNIST')

def create_and_train_model(train_ds_path ='', test_ds_path =''):
	'''
		Function that imports training and test datasets from user input or from EMNIST,
		creates the neural network and trains it
	'''
	# import user datasets or get EMNIST ones
	if train_ds_path == '':
		print('Take default training dataset from EMNIST...')
		training_samples, training_labels = md.get_training_dataset()
	else:
		try:
			print('Importing training dataset...')
			training_samples, training_labels = md.import_dataset(train_ds_path)
			print('Successful import of training dataset!')
		except:
			raise Exception("ERROR: file not found")

	if test_ds_path == "":
		print('Take default test dataset from EMNIST...')
		test_samples, test_labels = md.get_test_dataset()
	else:
		try:
			print('Importing test dataset...')
			test_samples, test_labels = md.import_dataset(test_ds_path)
			print('Successful import of test dataset!')
		except:
			print('Error: files not found!')

	# create model
	print("Creating and training model...")
	model = md.create_model(NUM_CLASSES)

	# define the optimizer
	#sgd = SGD(learning_rate=LEARNING_R, momentum=MOMENTUM, decay=DECAY_R, nesterov=False)

	# compile and train model
	model.compile(loss ="sparse_categorical_crossentropy", optimizer ='adam', metrics=["accuracy"])
	history = model.fit(training_samples, training_labels, batch_size= BATCH_SIZE, epochs= EPOCHS)
	score = model.evaluate(test_samples, test_labels, verbose=1)

	print('-' * 80)
	print('Model with batch size = {} and epochs = {}\n\tTest loss: {}\n\tTest accuracy: {}\n'.format(
		BATCH_SIZE,
		EPOCHS,
		score[0],
		score[1]
		)
	)

	save = ''
	while save != 'y' and save != 'Y' and save != 'n' and save != 'N':
		# save model
		save = input('Do you want to save this model? [y/n]')
		if save == 'y' or save == 'Y':
			model_filename = "nn_model.h5"
			model.save(os.path.join(MODELS_PATH, model_filename))
			print('Model saved!')
		elif save == 'n' or save == 'N':
			break
		else:
			print("Please insert y/Y for yes or n/N for no")
	return model


if __name__ == '__main__':
	args = parser.parse_args()
	create_and_train_model(str(args.train_dataset_path), str(args.test_dataset_path))