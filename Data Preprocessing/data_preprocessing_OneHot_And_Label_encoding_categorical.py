import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder

dataset = pd.read_csv("Data.csv")

missing_values = dataset.isna().sum()

print(f"Missing values: \n{missing_values}")

X_independent_variables = dataset.iloc[:, :-1].values
y_dependent_variable = dataset.iloc[:, -1].values

print(f"\nX independent variables: \n{X_independent_variables}\n")
print(f"\ny dependent variable: \n{y_dependent_variable}\n")

simple_imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
simple_imputer.fit(X_independent_variables[:, 1:3])
X_independent_variables[:, 1:3]=np.ceil(simple_imputer.transform(X_independent_variables[:, 1:3]))

print(f"\nReplaced X independent variables: \n{X_independent_variables}\n")

# ONE HOT ENCODING ~ Encoding caterogical data ~ Matrix of Features.
column_transformer = ColumnTransformer(transformers=[("encoder", OneHotEncoder(), [0])], remainder="passthrough")
column_transformer.fit(X_independent_variables)
X_independent_variables = column_transformer.transform(X_independent_variables)

print(f"After OneHot Encoding X independent variables: \n{X_independent_variables}\n")

# LABEL ENCODING ~ Encoding caterogical data for output/ dependent variable (direct encoding -> 0, 1 or 2)).
label_encoder = LabelEncoder()
label_encoder.fit(y_dependent_variable)
y_dependent_variable = label_encoder.transform(y_dependent_variable)

print(f"After Label Encoding y dependent variable: {y_dependent_variable}\n")