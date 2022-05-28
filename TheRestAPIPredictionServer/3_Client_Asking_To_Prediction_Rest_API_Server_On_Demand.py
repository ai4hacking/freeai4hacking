#AI4HACKING
#Created by Raúl Moreno Izquierdo
#Dataset: The MNIST (Modified National Institute of Standards and Technology) data consists of 60,000 training images and 10,000 test images. Each image is a crude 28 x 28 (784 pixels) handwritten digit from "0" to "9." Each pixel value is a grayscale integer between 0 and 255.

import argparse
from cgitb import text
import json
import requests
import numpy as np
from google.protobuf import json_format
from flask import Flask, render_template, request, flash,redirect
import tensorflow as tf
import tensorflow.math as math
import numpy as np
from flask_wtf import FlaskForm
from wtforms import StringField
from wtforms.validators import DataRequired
from flask_wtf.csrf import CSRFProtect
import os
from PIL import Image
import base64

SECRET_KEY = os.urandom(32)


app = Flask(__name__)

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

app.config['SECRET_KEY'] = SECRET_KEY

csrf = CSRFProtect(app)

class MyForm(FlaskForm):
    name = StringField('name', validators=[DataRequired()])


@app.route("/", methods=['GET', 'POST'])
def index():
 
    form = MyForm()

    if request.method == 'GET':
        return render_template("index.html",form=form)

    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file:
            filename = request.files['file'].read()
            #Encoding image as base64
            image_string = base64.b64encode(filename)
            image_string = image_string.decode('utf-8')
        
        #Creating json object
        payload = {
            "img": image_string
        }

        result = request.form.to_dict(flat=True)

        # sending post request to Rest API TensorFlow Serving server
        r = requests.post('http://localhost:9001/predict', json=payload)
        pred = json.loads(r.content.decode('utf-8'))
        pred = json.dumps(pred)
        result["msg"] = pred
        
        return render_template("index.html",form=form,result=result)

if __name__ == "__main__":
     app.run(host='0.0.0.0', port=9000, debug=True)