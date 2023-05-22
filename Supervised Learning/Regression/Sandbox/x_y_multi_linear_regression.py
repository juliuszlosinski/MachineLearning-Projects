import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

############# TRAINING DATA ############

training_input_data = np.array([
    [0, 1],
    [5, 1],
    [15, 2],
    [25, 5],
    [35, 11],
    [45, 15],
])
training_output_data = np.array([4, 5, 10, 15, 20, 30])

#########################################

############# VALIDATING DATA ###########

validating_input_data = training_input_data

##########################################


multi_linear_regression_model = LinearRegression()
multi_linear_regression_model.fit(training_input_data, training_output_data)

r_square = multi_linear_regression_model.score(training_input_data, training_output_data)
print(f"Coefficient of determination: {r_square}")
print(f"Intercept: {multi_linear_regression_model.intercept_}")
print(f"Cofficients: {multi_linear_regression_model.coef_}")

validation_predicted_output_data = multi_linear_regression_model.predict(validating_input_data)

print(f"Validation input data: {validating_input_data}\nOutput predicted data: {validation_predicted_output_data}")

validation_prediction_formule = multi_linear_regression_model.intercept_ + f" {multi_linear_regression_model.coef_}"