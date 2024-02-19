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
from sklearn.base import BaseEstimator
from sklearn.model_selection import (
    cross_validate,
    KFold,
)

def scale_data(X):
    X = X.astype(float)
    X = X / X.max()
    return X

def train_simple_classifier_with_cv(
    *,
    Xtrain: NDArray[np.floating],
    ytrain: NDArray[np.int32],
    clf: BaseEstimator,
    cv: KFold = KFold,
):
    """
    Train a simple classifier using k-vold cross-validation.

    Parameters:
        - X: Features dataset.
        - y: Labels.
        - cv_class: The cross-validation class to use.
        - estimator_class: The training classifier class to use.
        - n_splits: Number of splits for cross-validation.
        - print_results: Whether to print the results.

    Returns:
        - A dictionary with mean and std of accuracy and fit time.
    """
    scores = cross_validate(clf, Xtrain, ytrain, cv=cv, scoring='accuracy')
    return scores