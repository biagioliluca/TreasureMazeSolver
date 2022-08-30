from utils import model as md
from utils import NUM_CLASSES

import argparse
from pathlib import Path
import os

MODELS_PATH = os.path.join('..', 'models')

EPOCHS = 15
BATCH_SIZE = 64

parser = argparse.ArgumentParser()
parser.add_argument('--train_dataset_path', type=Path)
parser.add_argument('--test_dataset_path', type=Path)

if __name__ == '__main__':
	args = parser.parse_args()
	args.train_dataset_path
	if args.train_dataset_path is None and args.test_dataset_path is None:
		print('Take default dataset from EMNIST...')
		training_samples, training_labels, test_samples, test_labels = md.get_dataset()
	else:
		try:
			training_dataset_path = args.train_dataset_path
			test_dataset_path = args.test_dataset_path
			print('Importing training dataset...')
			training_samples, training_labels = md.import_dataset(training_dataset_path)
			print('Successful import of training dataset!\nImporting test dataset...')
			test_samples, test_labels = md.import_dataset(test_dataset_path)
			print('Successful import of test dataset!')
		except:
			print('Error: files not found!')
		
		else:
			# create and train nn-model
			print("Creating and training model...")
			model = md.create_model(NUM_CLASSES)
			model.compile(loss ="sparse_categorical_crossentropy",optimizer = "adam", metrics=["accuracy"])
			history = model.fit(training_samples, training_labels, batch_size= BATCH_SIZE, epochs= EPOCHS)
			score = model.evaluate(test_samples, test_labels, verbose=1)
			print('-', 80)
			print('Model with batch size = {} and epochs = {}\n\tTest loss: {}\n\tTest accuracy: {}\n'.format(
				BATCH_SIZE,
				EPOCHS,
				score[0],
				score[1]
				)
			)

			# save nn-model
			save = input('Do you want to save this model? [y/n]')
			if save == 'y' or save == 'Y':
				model_filename = input("Insert name file (include .h5): ")
				model.save(os.path.join(MODELS_PATH, model_filename))
				print('Model saved!')
			