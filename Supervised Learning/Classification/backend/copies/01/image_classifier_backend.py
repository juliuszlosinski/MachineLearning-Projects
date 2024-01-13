import cnn_models_handler as cnh
from flask import Flask, request
import firebase_storage_handler as fsh

##### LOADING MODELS #####

cnn_primitives_model = cnh.CNN_Model(
    model_name = "Primitives_classifier",
    path_to_model="./models_implementation/primitives_cnn_model.keras", 
    class_names={0: 'circle', 1: 'elipse', 2: 'rectangle', 3: 'square', 4: 'triangle'},
    trained_input_dimension=(64, 64)
)
cnn_primitives_model.load()

cnn_digits_model = cnh.CNN_Model(
    model_name = "Digits_classfier",
    path_to_model="./models_implementation/digits_cnn_model.keras",
    class_names={0:'0', 1:'1', 2:'2', 3:'3', 4:'4', 5:'5', 6:'6', 7:'7', 8:'8', 9:'9'},
    trained_input_dimension=(28, 28)
)
cnn_digits_model.load()

######## FIREABASE CONNECTION #####
firebase_handler = fsh.FirebaseHandler(path_to_config="./admin_key.json")
firebase_handler.connect()

######## REST API ########
app = Flask(__name__)

def allowed_file(file_name):
    return '.' in file_name and file_name.rsplit('.', 1)[1] in {'png', 'jpg', 'jpeg'}

# Default Web Page/ Home
@app.route("/")
def hello():
    return "<p>Image Classifier:</p> <p>/classify_primitive => Primitives</p><p>/classify_digit => Digits</p>"

@app.route("/classify", methods=["POST"])
def classify():
    """
    Needed input format:
    {
        "type": "primitive" or "digit",
        "label": if primitive: {"circle", "elipse", "rectangle", "square"} elif digit: {"1", "2", "3", "4", "5", "6", "7", "8", "9, "0"},
        "time": [0, infinity],
        "image": {"....png", "....jpg", "....jpeg"}
    }
    """
    type = request.form.get('type')
    label = request.form.get('label')
    time = request.form.get('time')
    image = request.files['image']
    
    path_to_image = "./last_image.png"
    if image and allowed_file(image.filename):
        image.save(path_to_image)
    else:
        return "Wrong format of image!"
    match type:
        case "primitive":
            prediction = cnn_primitives_model.predict(path_to_image)
            print(f"LOG::PRIMITIVE:PREDICTION = {prediction}")
        case "digit":
            prediction = cnn_primitives_model.predict(path_to_image)
            print(f"LOG::DIGIT::PREDICTION = {prediction}")
        case _:
            return "Wrong passed type!"
    if label != prediction["label"]:
        score = 0
    else:
        score = 100/float(time) * float(prediction["accuracy"])
    result = {
        "accuracy": prediction["accuracy"],
        "predicted_label": prediction["label"],
        "actual_label": label,
        "score": score
    }
    print(f"LOG::RESULT = {result}")
    return result

@app.route("/scores", methods=["POST"])
def add_score():
    """
    Adding score to Firebase database:
    data = {
        "accuracy": 25,
        "taskId": 2,
        "time":69,
        "userId": "rp0UWpYakLUWjLz84WPYsIQErwj1"
    }
    """
    accuracy = request.form.get("accuracy")
    taskId = request.form.get("taskId")
    time = request.form.get("time")
    userUid = request.form.get("userUid")
    score = {
        "accuracy": accuracy,
        "taskId": taskId,
        "time": time,
        "userUid": userUid,
    }
    print(f"LOG::ADD_SCORE = {score}")
    firebase_handler.post_document("scores", score)
    return "Score was added!"

if __name__ == '__main__':
    app.run(debug=True, port=5000)