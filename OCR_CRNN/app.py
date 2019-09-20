# Dependencies
from flask import Flask, request, jsonify
import joblib
import traceback
import numpy as np
import matplotlib.pyplot as plt
import os
import base64
from io import BytesIO

from check import working

app = Flask(__name__)

@app.route('/')
def hello():
    return 'TCS humAIn'

@app.route('/predict', methods=['POST'])
def predict():
    input_file = request.files.get('file')
    if not input_file:
	    return "File is not present in the request"
    if input_file.filename == '':
	    return "Filename is not present in the request"
    if not input_file.filename.lower().endswith(('.jpg', '.jpeg', '.png')):
	    return "Invalid file type"
    
    else:
        # input_buffer = BytesIO()
        # input_file.save(input_buffer)
        return working(input_file)
        # image_array = imread(input_buffer, as_gray=True)
        
        # category = evaluate_image(image_array)
        # return jsonify({'Category': str(category)})

if __name__ == '__main__':
    try:
        port = int(sys.argv[1]) # This is for a command-line input
    except:
        port = 8000 # If you don't provide any port the port will be set to 12345

    # model_columns = joblib.load("model_columns.pkl") # Load "model_columns.pkl"
    # print ('Model columns loaded')

    app.run(port=port, debug=True)
