"""
   Use to create your own functions for reuse 
   across the assignment

   Inside part_1_template_solution.py, 
  
     import new_utils
  
    or
  
     import new_utils as nu
"""
import numpy as np
from typing import Type, Dict
from numpy.typing import NDArray

def scale_data(X):
    X = X.astype(float)
    X = X / X.max()
    return X