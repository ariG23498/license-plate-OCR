
# To handle requests
import requests

# To randomise data
import random

# Linear algebra
import numpy as np

# data processing, CSV file I/O (e.g. pd.read_csv)
import pandas as pd 

# Image Processing
from skimage.io import imread
from skimage.filters import threshold_otsu
from skimage import measure
from skimage.measure import regionprops
from skimage.transform import resize
import cv2
from PIL import Image

# Plotting
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Handles url
import urllib

# For json support
import json

# For ML
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score

# For OS relates jobs
import os
import joblib

def working(input_buffer):
        gray_plate_image = imread(input_buffer, as_gray=True)
        # it should be a 2 dimensional array
        gray_plate_image = gray_plate_image * 255

        threshold_value = threshold_otsu(gray_plate_image) # thresholding
        binary_plate_image = gray_plate_image > threshold_value

        # The invert was done so as to convert the black pixel to white pixel and vice versa
        license_plate = (binary_plate_image)
        labelled_plate = measure.label(license_plate)

        # the next two lines is based on the assumptions that the width of
        # a license plate should be between 5% and 15% of the license plate,
        # and height should be between 35% and 60%
        # this will eliminate some
        # character_dimensions = (0.35*license_plate.shape[0], 0.60*license_plate.shape[0], 0.05*license_plate.shape[1], 0.15*license_plate.shape[1])
        character_dimensions = (0.25*license_plate.shape[0], 0.50*license_plate.shape[0], 0.05*license_plate.shape[1], 0.25*license_plate.shape[1])
        min_height, max_height, min_width, max_width = character_dimensions

        characters = []
        counter=0
        column_list = []
        for regions in regionprops(labelled_plate):
                y0, x0, y1, x1 = regions.bbox
                region_height = y1 - y0
                region_width = x1 - x0

                if region_height > min_height and region_height < max_height and region_width > min_width and region_width < max_width:
                        roi = license_plate[y0:y1, x0:x1]

                        # draw a red bordered rectangle over the character.
                        # rect_border = patches.Rectangle((x0, y0), x1 - x0, y1 - y0, edgecolor="red",linewidth=2, fill=False)


                        # resize the characters to 20X20 and then append each character into the characters list
                        resized_char = resize(roi, (20, 20))
                        characters.append(resized_char)

                        # this is just to keep track of the arrangement of the characters
                        column_list.append(x0)

        letters = [
            '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D',
            'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T',
            'U', 'V', 'W', 'X', 'Y', 'Z'
        ]


        # load the model
        current_dir = os.path.dirname(os.path.realpath('Implementation')) # Implementation refers to the file name
        model_dir = os.path.join(current_dir, 'models/svc/svc.pkl')
        model = joblib.load(model_dir)

        classification_result = []
        for each_character in characters:
                # converts it to a 1D array
                each_character = each_character.reshape(1, -1);
                result = model.predict(each_character)
                classification_result.append(result)

        print(classification_result)

        plate_string = ''
        for eachPredict in classification_result:
                plate_string += eachPredict[0]

        print(plate_string)

        # it's possible the characters are wrongly arranged
        # since that's a possibility, the column_list will be
        # used to sort the letters in the right order

        column_list_copy = column_list[:]
        column_list.sort()
        rightplate_string = ''
        for each in column_list:
                rightplate_string += plate_string[column_list_copy.index(each)]

        print(rightplate_string)
        return rightplate_string

