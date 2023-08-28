import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

# 1. Loading data and printing missing values.

dataset = pd.read_csv("titanic.csv")

missing_values = dataset.isna().sum()

print(f"\nMissing values:\n {missing_values}\n")

# 2. Spliting data to X independent variables/ Matrix of Features and y dependent variable (output).

X_independent_variables = dataset.drop(columns=["Survived"])
y_dependent_variable = dataset["Survived"]

print(f"X independent variables/ Matrix of features: \n{X_independent_variables}\n")
print(f"\ny dependent variable: {y_dependent_variable}\n")

# One Hot Encoding - X independent variables/ Matrix of Features

categorical_features = ["Sex", "Embarked", "Pclass"]
column_transformer = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), categorical_features)], remainder='passthrough')
column_transformer.fit(X_independent_variables)
X_independent_variables = column_transformer.transform(X_independent_variables)

print(f"\nNew X independent variables: \n{X_independent_variables}")

# Label Encoding - y dependent variable (output variable).
label_encoder = LabelEncoder()
label_encoder.fit(y_dependent_variable)
y_dependent_variable = label_encoder.transform(y_dependent_variable)

print(f"\n New y dependent variable: {y_dependent_variable}\n")