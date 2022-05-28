#AI4HACKING
#Created by Raúl Moreno Izquierdo
#Dataset: The MNIST (Modified National Institute of Standards and Technology) data consists of 60,000 training images and 10,000 test images. Each image is a crude 28 x 28 (784 pixels) handwritten digit from "0" to "9." Each pixel value is a grayscale integer between 0 and 255.

from flask import Flask, request
import tensorflow as tf
import tensorflow.math as math
import numpy as np
from wtforms import StringField
from wtforms.validators import DataRequired
import os
import cv2
from PIL import Image
from flask_cors import CORS
import base64
from flask import jsonify

SECRET_KEY = os.urandom(32)

app = Flask(__name__)

UPLOAD_FOLDER = '/tmp/images'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

new_model = tf.keras.models.load_model('ai4hacking/my_model_MNIST_AI4HACKING.h5')

@app.route("/predict", methods=['POST'])
def index():


    if request.method == 'POST':
        # POST request

        body = request.get_json()
        
        if 'img' not in body:
            response_body = {
                "msg": "img is null"
            }
            return jsonify(response_body), 200
        
        img = body["img"]

        with open(os.path.join(app.config['UPLOAD_FOLDER'])+'/test', 'w') as file:
            file.write(img)

        path = os.path.join(app.config['UPLOAD_FOLDER'])+'/test'
        
        nparr = np.fromstring(base64.b64decode(img), np.uint8) 
        image = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (28, 28),interpolation = cv2.INTER_NEAREST)
        image = image.astype('float32')
        image = image.reshape(1, 28, 28, 1)
        image = 255-image
        image /= 255

        #Make prediction
        pred = new_model.predict(image)
    
        #Create a json object for the response
        response_body = {
                "msg": str(np.argmax(pred)) + ": " + str(np.max(pred))
        }
        
        return jsonify(response_body), 200


if __name__ == "__main__":
     app.run(host='0.0.0.0', port=9001, debug=True)
      