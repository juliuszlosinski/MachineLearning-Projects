import numpy as np
from sklearn.datasets import load_iris

data_set = load_iris()

X_independet_variables = data_set.data[0:50, :]
y_dependent_variable = data_set.target[:]

