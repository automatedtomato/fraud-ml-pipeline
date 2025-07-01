import sys
from pathlib import Path
from constants import *

def setup_import():
    
    sys.path.append(ROOT)
    
    print(f"ROOT: {ROOT}")