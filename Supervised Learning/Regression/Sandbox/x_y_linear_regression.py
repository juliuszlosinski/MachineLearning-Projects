import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

################### TRAINING DATA ######################

training_x_data = np.array([5, 15, 25, 35, 45, 55]).reshape((-1, 1))
training_y_data = np.array([5, 20, 14, 32, 22, 28])

########################################################

################## VALIDATING DATA #####################

validating_x_data = training_x_data

########################################################

print(f"X: {training_x_data}, \nY: {training_y_data}")

linear_regression_model = LinearRegression()
linear_regression_model.fit(training_x_data, training_y_data)
r_square = linear_regression_model.score(training_x_data, training_y_data)

print(f"Coefficient of determination: {r_square}")
print(f"Intercept: {linear_regression_model.intercept_}")
print(f"Coefficients: {linear_regression_model.coef_}")

linear_regression_formula = f"f(x)= {linear_regression_model.coef_[0]} * {linear_regression_model.intercept_}"

print(linear_regression_formula)

y_predicted_values = linear_regression_model.predict(validating_x_data)

print(f"Predicted Y values: {y_predicted_values}")

plt.scatter(training_x_data, training_y_data, color="blue", marker='o', label="Default")
plt.plot(training_x_data, y_predicted_values, color="red", label=f"{linear_regression_formula}")
plt.xlabel("x [-]")
plt.ylabel("f(x) [-]")
plt.title(f"f(x) ~ y ~ Linear Regression")
plt.grid(True)
plt.legend()
plt.show()