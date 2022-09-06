# imports
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# constants
labels_table = {
    1: 0,   # 1
    2: 1,   # 2
    3: 2,   # 3
    4: 3,   # 4
    28: 4,  # S
    29: 5,  # T
    33: 6   # X
}

labels_to_digit = {
      1: '1',
      2: '2',
      3: '3',
      4: '4',
      28: 'S',
      29: 'T',
      33: 'X'
   }


NUM_CLASSES = len(labels_table)

grids = {
    'grid_2.png': [0, 2, 3, 0, 4, 1, 6, 2, 6, 3, 6, 1, 1, 2, 5, 0],
    'grid_3.png': [0, 4, 5, 6, 3, 2, 0, 3, 1, 6, 2, 0, 5, 1, 3, 2],
    'grid_5.png': [0, 3, 4, 5, 2, 1, 3, 6, 6, 0, 1, 0, 2, 5, 6, 3],
    'grid_6.png': [4, 0, 2, 3, 1, 0, 5, 6, 2, 6, 1, 0, 5, 3, 6, 2],
    'grid_7.png': [4, 0, 3, 2, 1, 0, 5, 6, 2, 6, 1, 0, 5, 3, 6, 2],
    'grid_8.png': [5, 0, 3, 1, 0, 2, 6, 2, 0, 3, 1, 4, 6, 1, 6, 0, 3, 2, 5, 5, 1, 0, 1, 6, 3]
    }

