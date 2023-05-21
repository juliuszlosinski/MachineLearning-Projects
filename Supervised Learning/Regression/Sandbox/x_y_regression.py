import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

################### LEARN DATA ######################

learn_x_data = np.array([5, 15, 25, 35, 45, 55]).reshape((-1, 1))
learn_y_data = np.array([5, 20, 14, 32, 22, 28])

#####################################################

################# VALIDATION DATA ###################

test_x_data = learn_x_data

#####################################################

print(f"X: {learn_x_data}, \nY: {learn_y_data}")

linear_regression_model = LinearRegression()
linear_regression_model.fit(learn_x_data, learn_y_data)
r_square = linear_regression_model.score(learn_x_data, learn_y_data)

print(f"Coefficient of determination: {r_square}")
print(f"Intercept: {linear_regression_model.intercept_}")
print(f"Coefficients: {linear_regression_model.coef_}")

linear_regression_formula = f"f(x)= {linear_regression_model.coef_[0]} * {linear_regression_model.intercept_}"

print(linear_regression_formula)

y_predicted_values = linear_regression_model.predict(test_x_data)

print(f"Predicted Y values: {y_predicted_values}")

plt.scatter(learn_x_data, learn_y_data, color="blue", marker='o', label="Default")
plt.plot(learn_x_data, y_predicted_values, color="red", label=f"{linear_regression_formula}")
plt.xlabel("x [-]")
plt.ylabel("f(x) [-]")
plt.title(f"f(x) ~ y ~ Linear Regression")
plt.grid(True)
plt.legend()
plt.show()