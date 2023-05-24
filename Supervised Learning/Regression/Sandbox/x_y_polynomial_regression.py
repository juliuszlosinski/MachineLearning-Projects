import numpy as np
import sklearn.linear_model as skl
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures

training_input_x_data = np.array([
    [0],
    [20],
    [40],
    [60],
    [80],
    [100]
])
training_output_y_data = np.array([0.02, 0.12, 0.6, 0.9, 1.8, 2.7])

print(f"Training input x data: {training_input_x_data}")
print(f"Training output y data: {training_output_y_data}")

linear_regreesion_model = skl.LinearRegression()
linear_regreesion_model.fit(training_input_x_data, training_output_y_data)

r_sqr = linear_regreesion_model.score(training_input_x_data, training_output_y_data)
intercept = linear_regreesion_model.intercept_
coefficients = linear_regreesion_model.coef_

print(f"R^2 = {r_sqr}")
print(f"Intercept = {intercept}")
print(f"Coefficients = {coefficients}")

predicted_values = linear_regreesion_model.predict(training_input_x_data)

plt.scatter(training_input_x_data, training_output_y_data, color="blue", label="default")
plt.plot(training_input_x_data, predicted_values, color="green", label=f"y = f(x) = {coefficients}*x + {intercept}")
plt.title("F(x) = y => Linear Regression Model")
plt.legend()
plt.grid(True)
plt.xlabel("x [-]")
plt.ylabel("y [-]")
plt.show()

poly_regression_model = PolynomialFeatures(degree=4)
X_poly = poly_regression_model.fit_transform(training_input_x_data)
poly_regression_model = skl.LinearRegression()
poly_regression_model.fit(X_poly, training_output_y_data)

r_sqr = poly_regression_model.score(X_poly, training_output_y_data)
intercept = poly_regression_model.intercept_
coefficients - poly_regression_model.coef_

print(f"R^2 = {r_sqr}")
print(f"Intercept = {intercept}")
print(f"Coefficients = {coefficients}")

predicted_output_values = poly_regression_model.predict(X_poly)

plt.scatter(training_input_x_data, training_output_y_data, color="blue")
plt.plot(training_input_x_data, predicted_output_values, color="green")
plt.xlabel("x [-]")
plt.ylabel("y [-]")
plt.grid(True)
plt.title("F(x) = y => Polynomial Regreesion model")
plt.show()