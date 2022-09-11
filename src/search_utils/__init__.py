import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(Path(__file__).resolve().parent.parent.parent, 'aima-python'))
print(sys.path)