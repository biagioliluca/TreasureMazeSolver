from nn_utils.digit_recognition import extract_and_preprocess 
from nn_utils import *
from nn_utils.train_model import create_and_train_model, MODELS_PATH

from search_utils.search_algorithms import *

import argparse
from pathlib import Path
import keras
import numpy as np
import math
from matplotlib import pyplot as plt


def get_value_from_label(table, value):
	'''
	Function that convert the label back to the original char value
	'''
	keys = list(table.keys())
	values = list(table.values())
	i = values.index(value)
	return labels_to_digit[keys[i]]

def find_start(grid):
	''''
		Finds start state 
	'''
	start_found = False
	initial_state = (-1,-1)
	for i in range(len(grid)):
		for j in range(len(grid[i])):
			if grid[i][j] == 'S':
				if start_found:
					raise Exception('ERROR! Found more than 1 start point: you can only have ONE start point')
				initial_state = (i,j) 
				start_found = True
		
	if initial_state == (-1,-1):
		raise Exception('ERROR: There is no start point')
	return initial_state
		
if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-p', '--path', type=Path, help='specify the path to the specific grid')
	parser.add_argument('number_of_treasures', type=int, help='specify the number of treasures you want to find')
	parser.add_argument('-a', '--algorithm', type=int, choices=[0, 1],  help='specify the search algorithm you want to use:\n- 0: dijkstra\n- 1: A star')
	args = parser.parse_args()

	# 1. take an image as input
	try:
		image_name = str(args.path)
		if image_name == None:
			raise Exception('Please choose a grid')
	except:
		raise Exception('There is no path as {}'.format(args.p))

	# 2. convert the image into a numeric matrix
	print('Extracting digits from grid...')
	digits, image_value = extract_and_preprocess(image_name)

	# 3. load the model or create a new one
	create_new_model = ''
	while create_new_model != 'y' and create_new_model != 'Y' and create_new_model != 'n' and create_new_model != 'N':
		create_new_model = input('Do you want to create a new model? [y/n]')
		if create_new_model == 'y' or create_new_model == 'Y':
			dataset_found_flag = False
			while dataset_found_flag == False:
				print('Please insert dataset paths if you want to import them, otherwise leave it blank')
				input_train_ds = input('Insert training dataset path: ')
				input_test_ds = input('Insert test dataset path: ')
				try:
					model = create_and_train_model(input_train_ds, input_test_ds)
					print('Model created!')
					dataset_found_flag = True
					break
				except:
					print('Error: file dataset not found, try again')

		elif create_new_model == 'n' or create_new_model == 'N':
			model = keras.models.load_model(os.path.join(MODELS_PATH, "nn_model.h5"))
			break
		else:
			print('Please insert y/Y for yes or n/N for no')

	# 4. create the grid with predicted digits
	predicted = []
	for i in range(len(digits)):    
		predict_digit = model.predict(digits[i:i+1])
		class_digit = np.argmax(predict_digit,axis=1)
		predicted.append(get_value_from_label(labels_table, class_digit))

		# print image with relative predicted value
		#plt.imshow(image_value[i])
		#plt.show()
		#print(class_digit)

	n = int(math.sqrt(len(digits)))
	grid = []

	for i in range(n):
		row =[]
		for j in range(n):
			row.append(predicted[(i*n)+j])
		grid.append(row)

	# 5. resolve the problem using a* or dijkstra
	problem_maze = TreasureMazeProblem(find_start(grid), grid, args.number_of_treasures)
	if args.algorithm:
		solution = solve_treasure_maze_a_star(problem_maze)
	else:
		solution = solve_treasure_maze_dijkstra(problem_maze)
	string = ''
	flag=True
	for sub_list in solution:
  		if flag == False:
  			string += '->'
  		string += sub_list[0]
  		flag=False
  		for action in sub_list[1:]:
  			string += '->' + action

	print(string)  
