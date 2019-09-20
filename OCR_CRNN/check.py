import cv2
import numpy as np
from ocrdetector import recognize
from skimage.io import imread


# detector = OCRdetector('weights/shadownet.ckpt', 'data/char_dict/char_dict.json', 'data/char_dict/ord_map.json')


# image = cv2.imread('data/test_images/26.jpg', cv2.IMREAD_COLOR)
def working(image_file):

    image = imread(image_file)
    # image.save("check_img".jpg)
    # image = cv2.imread("", cv2.IMREAD_COLOR)


    return (recognize(image,'weights/shadownet.ckpt', 'data/char_dict/char_dict.json', 'data/char_dict/ord_map.json'))




