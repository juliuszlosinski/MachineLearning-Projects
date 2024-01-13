from keras.models import Sequential
from keras.layers import MaxPooling2D, Conv2D, BatchNormalization, Dropout, Activation, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
from keras.preprocessing.image import load_img
import numpy as np

number_of_channels = 1
input_image_width = 64
input_image_height = 64
input_image_shape = (input_image_width, input_image_height, number_of_channels)

cnn_model = Sequential()
cnn_model.add(Conv2D(32,(3,3),input_shape=input_image_shape))
cnn_model.add(Activation('relu'))
cnn_model.add(BatchNormalization())
cnn_model.add(MaxPooling2D(pool_size =(2,2)))
cnn_model.add(Conv2D(32,(3,3)))
cnn_model.add(Activation('relu'))
cnn_model.add(BatchNormalization())
cnn_model.add(MaxPooling2D(pool_size =(2,2)))
cnn_model.add(Flatten())
cnn_model.add(Dense(128))
cnn_model.add(Activation('relu'))
cnn_model.add(Dropout(0.5))
cnn_model.add(Dense(5))
cnn_model.add(Activation('softmax'))
cnn_model.summary()
cnn_model.compile(optimizer ='adam',
                   loss ='categorical_crossentropy',
                   metrics =['accuracy'])
training_data_generator = ImageDataGenerator(rescale =1./255,
                                   shear_range =0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip =True)
testing_data_generator = ImageDataGenerator(rescale = 1./255)

batchsize=16
training_data = training_data_generator.flow_from_directory("./training_data",
                                                target_size=(input_image_width, input_image_height),
                                                batch_size= batchsize,
                                                color_mode='grayscale',
                                                class_mode='categorical')

validation_data = testing_data_generator.flow_from_directory("./validation_data",
                                           target_size = (input_image_width, input_image_height),
                                           batch_size = batchsize,
                                           shuffle=False,
                                           color_mode='grayscale',
                                           class_mode ='categorical')

class_names = {value: key for key, value in training_data.class_indices.items()}

print(f"Classes: {class_names}")

cnn_model.fit(training_data,
              validation_data =validation_data,
                epochs=75,
                verbose=2,
)

cnn_model.save("./primitives_cnn_model.keras")
cnn_model.save("../../models_ready_to_use/primitives_cnn_model.keras")
cnn_model.save_weights("./primitives_cnn_model_weights.h5")
cnn_model.save_weights("../../models_ready_to_use/primitives_cnn_model_weights.h5")

# TESTING
cnn_loaded_model = load_model("./primitives_cnn_model.keras")
image = load_img("./validation_data/square/square60.png", target_size=(64, 64), color_mode="grayscale")
img = np.array(image)
img = img/255.0
img = img.reshape(1, 64, 64, number_of_channels)
output_label = cnn_loaded_model.predict(img)
print(f"Object: {class_names[np.argmax(output_label)]}")