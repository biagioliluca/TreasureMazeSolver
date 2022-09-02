from nn_utils import model as md
from nn_utils import NUM_CLASSES

import argparse
from pathlib import Path
import os

MODELS_PATH = os.path.join(Path(__file__).resolve().parent.parent, 'models')

EPOCHS = 15
BATCH_SIZE = 64

parser = argparse.ArgumentParser()
parser.add_argument('--train_dataset_path', type=Path, help="enter training dataset path, or leave it blank to extract it from EMNIST")
parser.add_argument('--test_dataset_path', type=Path, help="enter test dataset path, or leave it blank to extract it from EMNIST")

if __name__ == '__main__':
	args = parser.parse_args()
	args.train_dataset_path
	if args.train_dataset_path is None:
		print('Take default training dataset from EMNIST...')
		training_samples, training_labels = md.get_training_dataset()
	else:
		training_dataset_path = args.train_dataset_path
		try:
			print('Importing training dataset...')
			training_samples, training_labels = md.import_dataset(training_dataset_path)
			print('Successful import of training dataset!')
		except:
			print('Error: files not found!')

	if args.test_dataset_path is None:
		print('Take default test dataset from EMNIST...')
		test_samples, test_labels = md.get_test_dataset()
	else:
		test_dataset_path = args.test_dataset_path
		try:
			print('Importing test dataset...')
			test_samples, test_labels = md.import_dataset(test_dataset_path)
			print('Successful import of test dataset!')
		except:
			print('Error: files not found!')

		
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

	while save != 'y' and save != 'Y' and save != 'n' and save != 'N':
		# save nn-model
		save = input('Do you want to save this model? [y/n]')
		if save == 'y' or save == 'Y':
			model_filename = "nn_model.h5"
			model.save(os.path.join(MODELS_PATH, model_filename))
			print('Model saved!')
			#break
		elif save == 'n' or save == 'N':
			break
		else:
			print("Please insert y/Y for yes or n/N for no")