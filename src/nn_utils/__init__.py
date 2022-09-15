# imports
import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

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
