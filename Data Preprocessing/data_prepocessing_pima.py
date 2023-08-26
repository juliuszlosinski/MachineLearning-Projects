import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

dataset = pd.read_csv("pima-indians-diabetes.csv")

missing_data = dataset.isnull().sum()

print(f"\nMissing data: \n{missing_data}\n")

X_independent_variables = dataset.iloc[:, :-1].values
y_dependent_variable = dataset.iloc[:, -1].values

print(f"\nX independent variables: \n{X_independent_variables}")
print(f"\ny dependent variable: \n{y_dependent_variable}")

# WARNING - Every feature is numerical attribute!

imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
imputer.fit(X_independent_variables)
X_independent_variables = imputer.transform(X_independent_variables)

print(f"\nX independent variables after replacing missing values: \n{X_independent_variables}")