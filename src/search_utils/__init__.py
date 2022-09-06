import os
from pathlib import Path
import sys
import inspect

sys.path.insert(0, os.path.join(Path(__file__).resolve().parent.parent.parent, 'aima-python'))
print(sys.path)