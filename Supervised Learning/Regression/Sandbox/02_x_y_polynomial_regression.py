import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

training_input_x_data = np.array([
    5, 10, 15, 20, 35, 40
])
training_output_y_data = np.array([
    20, 30, 40, 35, 20, 15
])

validation_input_x_data = training_input_x_data

poly = PolynomialFeatures(degree=2, include_bias=False)
poly_features = poly.fit_transform(training_input_x_data.reshape(-1, 1))

poly_regression_model = LinearRegression()
poly_regression_model.fit(poly_features, training_output_y_data)

predicted_output_y_data = poly_regression_model.predict(poly_features)

print(predicted_output_y_data)