from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import BatchNormalization
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Activation
from keras.layers import Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from keras.losses import CategoricalCrossentropy
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau
import numpy as np
import tensorflow as tf

image_width = 64
image_height = 64
batch_size = 16
number_of_channels = 1

training_data_directory_path = "training_data"
validation_data_directory_path = "validation_data"

training_image_data_generator = ImageDataGenerator(
    rescale=1.0/255,
    zoom_range=0.2,
    width_shift_range=0.2,
    height_shift_range=0.2,
    fill_mode='nearest'
)
validation_image_data_generator = ImageDataGenerator(
    rescale=1.0/255
)

traning_data_generator = training_image_data_generator.flow_from_directory(
    training_data_directory_path,
    target_size=(image_width, image_height),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True,
    color_mode ="grayscale" # FIRST SOLUTION
)
validation_data_generator = validation_image_data_generator.flow_from_directory(
    validation_data_directory_path,
    target_size=(image_width, image_height),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True,
    color_mode ="grayscale" # FIRST SOLUTION
)

class_names = {value: key for key, value in traning_data_generator.class_indices.items()}

print("Classes: ")
for key, value in class_names.items():
    print(f"{key}: {value}")
    
cnn_model = Sequential()

# FIRST LAYER
cnn_model.add(Convolution2D(
    filters=128,
    kernel_size=(5, 5),
    padding="valid",
    input_shape=(image_width, image_height, 1)
))
cnn_model.add(Activation(activation='relu'))
cnn_model.add(MaxPooling2D(pool_size=(2, 2)))
cnn_model.add(BatchNormalization())

# SECOND LAYER
cnn_model.add(Convolution2D(
    filters=64,
    kernel_size=(3, 3),
    padding='valid',
    kernel_regularizer=l2(0.00005)
))
cnn_model.add(Activation(activation='relu'))
cnn_model.add(MaxPooling2D(pool_size=(2, 2)))
cnn_model.add(BatchNormalization())

# THIRD LAYER
cnn_model.add(Convolution2D(
    filters=32,
    kernel_size=(3, 3),
    padding='valid',
    kernel_regularizer=l2(0.00005)
))
cnn_model.add(Activation('relu'))
cnn_model.add(MaxPooling2D(pool_size=(2, 2))) 
cnn_model.add(BatchNormalization())

cnn_model.add(Flatten())
cnn_model.add(Dense(units=256, activation='relu'))
cnn_model.add(Dropout(0.5))
cnn_model.add(Dense(units=5, activation='softmax'))

cnn_model.summary()

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=np.sqrt(0.1),
    patience=5
)

optimizer = Adam(learning_rate=0.001)

cnn_model.compile(
    optimizer=optimizer,
    loss=CategoricalCrossentropy(),
    metrics=['accuracy']
)

history = cnn_model.fit(
    traning_data_generator,
    epochs=150,
    verbose=2,
    validation_data=validation_data_generator,
    callbacks=[reduce_lr]
)

testing_data_directory_path = validation_data_directory_path
testing_image_data_generator = ImageDataGenerator(
    rescale=1.0/255
)
testing_data_generator = testing_image_data_generator.flow_from_directory(
    testing_data_directory_path,
    target_size=(image_width, image_height),
    shuffle=False,
    batch_size=batch_size,
    class_mode="categorical",
    color_mode ="grayscale" # FIRST SOLUTION
)

predictions = cnn_model.predict(
    testing_data_generator
)

test_loss, test_accuracy = cnn_model.evaluate(
    testing_data_generator, batch_size=batch_size
)

print(f"Test loss: {test_loss}")
print(f"Test accuracy: {test_accuracy}")

cnn_model.save_weights("primitives_cnn_model_weights.h5")
cnn_model.save("primitives_cnn_model.keras")

from keras.models import load_model
from keras.preprocessing.image import load_img

# Loading CNN model.
cnn_model = load_model("primitives_cnn_model.keras")


# Loading square
image = load_img("./validation_data/circle/circle-bad.png", target_size=(64, 64), color_mode="grayscale")
img = np.array(image)
img = img/255.0
img = img.reshape(1, 64, 64, number_of_channels)

output_label = cnn_model.predict(img)
print(output_label)
print(np.max(output_label))
print(f"Object: {class_names[np.argmax(output_label)]}")