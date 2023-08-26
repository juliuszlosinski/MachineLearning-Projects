import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt

dataset = pd.read_csv("Data.csv")

missing_values = dataset.isnull().sum()

print(f"\nMissing values: {missing_values}\n")

X_independent_variables = dataset.iloc[:, :-1].values
y_dependent_variable = dataset.iloc[:, -1].values

print(f"\nX independent variables: \n{X_independent_variables}\n")
print(f"\ny dependent variable: \n{y_dependent_variable}")

simple_imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
simple_imputer.fit(X_independent_variables[:, 1:3])
X_independent_variables[:, 1:3] = simple_imputer.transform(X_independent_variables[:, 1:3])
X_independent_variables[:, 1:3]=np.ceil(X_independent_variables[:, 1:3])

print(f"\nNew X independent variables: \n{X_independent_variables}\n")

age = X_independent_variables[:, 1]
salary = X_independent_variables[:, 2]

print(f"\nAge: {age}\n")
print(f"\nSalary: {salary}\n")

plt.scatter(age, salary)
plt.xlabel("Age [year]")
plt.ylabel("Salary [$]")
plt.title("Salary(age) [$]")
plt.grid(True)
plt.show()