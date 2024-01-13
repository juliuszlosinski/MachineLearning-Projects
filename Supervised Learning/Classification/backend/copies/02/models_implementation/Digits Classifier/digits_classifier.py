from keras.datasets import mnist
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import Activation
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Softmax
from keras.models import load_model
from keras.preprocessing.image import load_img
import numpy as np

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(60000, 28, 28, 1)
X_test = X_test.reshape(10000, 28, 28, 1)

print(f"Shape of X_train: {X_train.shape}")
print(f"Shape of X_test: {X_test.shape}")

for i in range(5):
    print(f"y_train[{i}]: \n{y_train[i]}")

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

for i in range(5):
    print(f"y_train[{i}]: {y_train[i]}")
    
cnn_model = Sequential()
cnn_model.add(Convolution2D(
    64, kernel_size=3, input_shape=(28, 28, 1)
))
cnn_model.add(Activation(activation="relu"))
cnn_model.add(Convolution2D(
    32, kernel_size=3
))
cnn_model.add(Activation(activation="relu"))
cnn_model.add(Flatten())
cnn_model.add(Dense(10))
cnn_model.add(Softmax())

cnn_model.compile(
    optimizer="adam", loss="categorical_crossentropy", metrics=['accuracy']
)

cnn_model.fit(
    X_train, y_train, validation_data=(X_test, y_test), epochs=3
)

cnn_model.save("digits_cnn_model.keras")
cnn_model.save_weights("digits_cnn_model_weights.h5")
number_of_channels=1

# Loading CNN model.
cnn_model = load_model("digits_cnn_model.keras")

# Loading square
image = load_img("./testing_data/1.png", target_size=(28, 28), color_mode="grayscale")
img = np.array(image)
img = img/255.0
img = img.reshape(1, 28, 28, number_of_channels)

output_label = cnn_model.predict(img)
print(f"Outputp label: {output_label}")