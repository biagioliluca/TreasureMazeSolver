from digit_recognition import *
from search_algorithms import *
import argparse
#from pathlib import Path
import sys
sys.path.append('../aima-python')
import search
import __init__
print(labels_to_digit[0])


def get_value_from_label(table, value):

  keys = list(table.keys())
  values = list(table.values())
  i = values.index(value)
  return labels_to_digit[keys[i]]

def find_start(grid):
  start_found = False
  initial_state = (-1,-1)
  for i in range(len(grid)):
    for j in range(len(grid[i])): 

      if grid[i][j] == 'S':

        if start_found:
          raise Exception("ERROR! Found more than 1 start point: you can only have ONE start point")
        initial_state = (i,j) 
        start_found = True

  return initial_state

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("-p", "--path", type=Path, help="specify the path to the specific grid")
	parser.add_argument("number_of_treasures", type=int, help="specify the number of treasures you want to find")
	parser.add_argument("-a", "--algorithm", type=int, choices=[0, 1],  help="specify the search algorithm you want to use:\n- 0: dijkstra\n- 1: A star")
	args = parser.parse_args()

	# 1. prendere in input un'immagine
	try:
		image_name = args.p
		if args.p == None:
			raise Exception("Please choose a grid")
	except:
		raise Exception("There is no path as {}".format(args.p))

	# 2. convertire tale immagine in una matrice numerica
	digits, _ = extract_and_preprocess(image_name)

	# 3. caricare il modello
	loaded_model = keras.models.load_model(train_model.save_dataset_path + "char_recognition_model.h5")

	# 4. creare la griglia dei valori predetti
	predicted = []
	for i in range(len(digits)):    
	    predict_digit = model.predict(digits[i:i+1])
	    class_digit = np.argmax(predict_digit,axis=1)
	    predicted.append(get_value_from_label(table_labels, class_digit))

	n = int(math.sqrt(len(digits)))
	grid = []

	for i in range(n):
	  row =[]
	  for j in range(n):
	    row.append(predicted[(i*n)+j])
	  grid.append(row)

	# 5. risolvere il problema
	problem_maze = TreasureMazeProblem(find_start(grid), grid, args.number_of_treasures)

	if args.a:
		solution = solve_treasure_maze_a_star(problem_maze, calculate_heuristic_grid_b(problem_maze))
	else:
		solution = solve_treasure_maze_dijkstra(problem_maze)

	print(solution)