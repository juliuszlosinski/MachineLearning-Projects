from flask import Flask, request
from keras.models import load_model
from keras.preprocessing.image import load_img
import numpy as np
import os 
import shutil

UPLOAD_FOLDER = 'sended_images'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}

######## MODELS ########

# 1. Loading Model for primitives classification.
cnn_primitives_model = load_model('cnn_model.keras')
cnn_primitives_labels = {0: 'circle', 1: 'elipse', 2: 'rectangle', 3: 'square', 4: 'triangle'}

def temporary_softmax(z):
    s = np.max(z, axis=1)
    s = s[:, np.newaxis] # necessary step to do broadcasting
    e_x = np.exp(z - s)
    div = np.sum(e_x, axis=1)
    div = div[:, np.newaxis] # dito
    return e_x / div

def predict_primitive(path_to_file, cnn_model, cnn_labels):
    image = load_img(path_to_file, target_size=(64, 64))
    img = np.array(image)
    img = img / 255.0
    img = img.reshape(1, 64, 64, 3)
    output_label = cnn_model.predict(img)
    # TEMPORARY SOFTMAX
    result_softmax = temporary_softmax(output_label)
    print(result_softmax)
    return cnn_labels[np.argmax(output_label)]

######## REST API ########

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(file_name):
    return '.' in file_name and file_name.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

# Default Web Page/ Home
@app.route("/")
def hello():
    return "<p>Image Classifier:</p> <p>/classify_primitive -> Primitives</p><p>/classify_alphabet -> Alphabet</p>"

# End point for classification primitives
@app.route('/classify_primitive', methods=['POST'])
def classify_primitive():
        file = request.files['file']
        if file and allowed_file(file.filename):
            print(file.filename)
            file.save(file.filename)
            print("File succefully uploaded!")
            predicted_label = predict_primitive(f"./{file.filename}", cnn_primitives_model, cnn_primitives_labels)
            shutil.copyfile(f"./{file.filename}", f"./last_image-{file.filename}")
            os.remove(f"./{file.filename}")
            return predicted_label
        else:
            return "Missing file!"