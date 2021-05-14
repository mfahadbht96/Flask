#!/usr/bin/env python


# Libraries
import cv2
import numpy as np
import os
import pickle as pk


# Flask utils
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)

# Settings
global pca
global model

#Loading PCA
pca=pk.load(open("pca.pkl",'rb'))

#Loading model
model=pk.load(open("model.pkl",'rb'))



@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        print(basepath)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        print(file_path)
        f.save(file_path)
        
        #process image file
        img = cv2.imread(file_path)  
        image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        image=image.reshape(1,784)
        img_pca=pca.transform(image)
        
        # Make prediction
        pred = model.predict(img_pca)

        # Process your result for human      
        result = pred[0]               # extract value
        return result
    return None

if __name__ == '__main__':
    app.run(debug=True)








