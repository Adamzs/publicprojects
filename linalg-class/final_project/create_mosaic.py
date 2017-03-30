import numpy as np
import cv2
import sys
import os
from PIL import Image

def eigen_train_data():
    face_dir = 'face_data2'
    resized_width, resized_height = (92*5, 112*4)

    new_im = Image.new('L', (resized_width, resized_height))
    for (subdirs, dirs, files) in os.walk(face_dir):
        for subdir in dirs:
            img_path = os.path.join(face_dir, subdir)
            row_count = 0
            col_count = 0
            x_offset = 0
            y_offset = 0
            for fn in os.listdir(img_path):
                path = img_path + '/' + fn
                #gray = cv2.imread(path, 0)
                gray = Image.open(path)
                xoff = x_offset + (col_count * 92)
                yoff = y_offset + (row_count * 112)
                new_im.paste(gray, (xoff,yoff))
                print str(xoff) + ", " + str(yoff)
                col_count += 1
                if col_count > 4:
                    col_count = 0
                    row_count += 1
                if row_count > 3:
                    new_im.save('test.jpg')
                    return
    new_im.save('test2.jpg')


eigen_train_data()

