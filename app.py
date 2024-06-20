from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
import tensorflow as tf
import numpy as np
import os
from PIL import Image


app = Flask(__name__)
model = tf.keras.models.load_model("1.keras")
class_names = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']
IMAGE_SIZE = 256
BATCH_SIZE = 32
CHANNELS = 3
EPOCHS = 50

def predict(img):
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array,0)
    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = round(100 * (np.max(predictions[0])), 2)
    return predicted_class, confidence



@app.route('/', methods=['POST', 'GET'])

def home():
    if request.method== 'POST':
        if "file" not in request.files:
            return render_template('index.html', message="No file found")
        file = request.files['file']

        if file.filename == '':
            return render_template('index.html', message="No file selected")
        
        if file and allow_file(file.filename):
            filename = secure_filename(file.filename)
            filepath= os.path.join('static', filename)
            file.save(filepath)

            img = tf.keras.preprocessing.image.load_img(filepath, target_size = (IMAGE_SIZE, IMAGE_SIZE))
            predicted_class, confidence = predict(img)

            # render the template with the uploaded image, actual and predicted labels, confidence
            return render_template('index.html', image_path=filepath, actual_label = predicted_class,
                                   predicted_label=predicted_class, confidence=confidence)
            



    return render_template('index.html')

def allow_file(filename):
    return '.' in  filename and filename.rsplit('.', 1)[1].lower() in {'png','jpeg','jpg'}

# python main 
if __name__ == "__main__":
    app.run(debug=True)