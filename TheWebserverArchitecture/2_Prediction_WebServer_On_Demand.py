#AI4HACKING
#Created by Raúl Moreno Izquierdo
#Dataset: The MNIST (Modified National Institute of Standards and Technology) data consists of 60,000 training images and 10,000 test images. Each image is a crude 28 x 28 (784 pixels) handwritten digit from "0" to "9." Each pixel value is a grayscale integer between 0 and 255.

from flask import Flask, render_template, request, make_response, flash, redirect
from flask_wtf import FlaskForm
from wtforms import StringField
from wtforms.validators import DataRequired
from flask_wtf.csrf import CSRFProtect
import os
from werkzeug.utils import secure_filename
import cv2
import numpy as np
import tensorflow as tf

app = Flask(__name__)

SECRET_KEY = os.urandom(32)
app.config['SECRET_KEY'] = SECRET_KEY

UPLOAD_FOLDER = '/tmp/images'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

csrf = CSRFProtect(app)

#Load the TensorFlow model, for predicting
new_model = tf.keras.models.load_model('my_model_MNIST_AI4HACKING.h5')

class MyForm(FlaskForm):
    name = StringField('name', validators=[DataRequired()])

    #Simple method for showing a web page interacting with the model
    @app.route("/", methods=['GET', 'POST'])
    def index():
    
        form = MyForm()

        if request.method == 'GET':
            return render_template("index.html",form=form)

        if request.method == 'POST':
            
            if 'file' not in request.files:
                flash('No file part')
                return redirect(request.url)
            file = request.files['file']
            if file.filename == '':
                flash('No selected file')
                return redirect(request.url)
            if file:
                filename = secure_filename(file.filename)
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], file.filename))
                
                result = request.form.to_dict(flat=True)
                
                #Read the image with opencv
                image = cv2.imread(os.path.join(app.config['UPLOAD_FOLDER'], file.filename), cv2.IMREAD_GRAYSCALE)
                
                #Resize and codify the sample to meet the original model requirements
                image = cv2.resize(image, (28, 28),interpolation = cv2.INTER_NEAREST)
                image = image.astype('float32')
                image = image.reshape(1, 28, 28, 1)
                image = 255-image
                image /= 255

                #Perform a prediction with the sample image
                pred = new_model.predict(image)

                #Prepare ouput selecting the most probable digit as the prediction
                result["msg"] = str(np.argmax(pred)) + ": " + str(np.max(pred))
            
            response = make_response(render_template("index.html", form=form,result=result))
            response.headers['Content-Security-Policy'] = "default-src 'self'"
            return response


if __name__ == "__main__":
     app.run(host='0.0.0.0', port=8000, debug=True)
      