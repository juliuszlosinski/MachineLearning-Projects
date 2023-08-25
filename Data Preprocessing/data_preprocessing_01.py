import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer 

dataset = pd.read_csv("Data.csv")

X_independent_variables = dataset.iloc[:, :-1].values # Matrix of features/ independent variables.
y_dependent_variable = dataset.iloc[:, -1].values # Dependent variable (f(x1, ..., xn)).

print(f"Independent variables: {X_independent_variables}")
print(f"Dependent variable: {y_dependent_variable}")

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(X_independent_variables[:, 1:3])
X_independent_variables[:, 1:3] = imputer.transform(X_independent_variables[:, 1:3])

print(X_independent_variables)