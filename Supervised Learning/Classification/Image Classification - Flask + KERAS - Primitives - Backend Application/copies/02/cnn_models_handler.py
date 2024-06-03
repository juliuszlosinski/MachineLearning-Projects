from keras.preprocessing.image import load_img
from keras.models import load_model
import numpy as np

"""
CNN_Model - Handling and loading Convolutioanl Neural Networks
Needs as input:
- path_to_model ~ path to file that contains KERAS model,
- class_names ~ dictionary of class names of objects in correct order,
- trained_input_dimension ~ dimension of images that were used for model training.
"""
class CNN_Model:
    def __init__(self, model_name, path_to_model, class_names, trained_input_dimension):
        """
        Initializng fundamentals fields.
        """
        self.path_to_log_file = f"./logs/{model_name}.log"
        self.path_to_model = path_to_model
        self.class_names = class_names
        self.trained_input_dimension = trained_input_dimension
        
    def load(self)->None:
        """
        Loading saved model.
        """
        self.cnn_model = load_model(self.path_to_model)
    
    def format_file(self, path_to_file:str):
        """
        Formatting file/ image to grayscale format.
        """
        number_of_channels=1
        image = load_img(path_to_file, target_size=self.trained_input_dimension, color_mode="grayscale")
        img = np.array(image)
        img = img/255.0
        img = img.reshape(1, self.trained_input_dimension[0], self.trained_input_dimension[1], number_of_channels)
        return img
    
    def predict(self, path_to_file:str)->dict:
        """
        Making prediction and returning accuracy and label.
        """
        file = open(f"{self.path_to_log_file}", "a")
        img = self.format_file(path_to_file)
        output = self.cnn_model.predict(img)
        accuracy = str(np.max(output))
        label = str(self.class_names[np.argmax(output)])
        file.write(f"Last prediction: {output}, Accuracy: {accuracy}, Label: {label}\n")
        file.close()
        return{"accuracy": accuracy, "label": label}