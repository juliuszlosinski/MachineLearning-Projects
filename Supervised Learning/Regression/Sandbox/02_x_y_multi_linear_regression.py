import numpy as np
import sklearn.linear_model as skl
import matplotlib.pyplot as plt

training_input_data = np.array([
    2, 3,
    4, 4,
    5, 6,
    8, 10,
    9, 12,
    10, 15,
])

training_output_data = np.array([
    5, 
    10,
    8,
    15,
    10,
    7
])

testing_input_data = training_input_data

i=0
training_x1_input_data, training_x2_input_data = [], []
while i < len(training_input_data):
    if i % 2 == 0:
        training_x1_input_data.append(training_input_data[i]) 
    else:
        training_x2_input_data.append(training_input_data[i])
    i+=1

print(f"\nX1: {training_x1_input_data}\nX2: {training_x2_input_data}\nY: {list(training_output_data)}")

multi_linear_regression_model = skl.LinearRegression()
multi_linear_regression_model.fit(training_input_data.reshape((-1, 2)), training_output_data)
predicted_output_values = multi_linear_regression_model.predict(training_input_data.reshape((-1, 2)))

print(f"Training input data: {training_input_data}")

print(f"Predicted values: {predicted_output_values}\n")

ax = plt.axes(projection ="3d")
ax.scatter3D(training_x1_input_data, training_x2_input_data,training_output_data,  color = "green")
ax.plot(training_x1_input_data, training_x2_input_data, predicted_output_values)
ax.set_xlabel("x1", fontweight ='bold')
ax.set_ylabel("x2", fontweight ='bold')
ax.set_zlabel("y", fontweight ='bold')
plt.title("Multiline Regression")
plt.savefig("multi_linear_regression.png")
plt.show()