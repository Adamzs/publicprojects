import numpy as np
import cv2
import sys
import os

RESIZE_FACTOR = 4

def eigen_train_data():
    face_dir = 'face_data'
    resized_width, resized_height = (92, 112)

    for (subdirs, dirs, files) in os.walk(face_dir):
        for subdir in dirs:
            img_path = os.path.join(face_dir, subdir)
            for fn in os.listdir(img_path):
                path = img_path + '/' + fn
                gray = cv2.imread(path, 0)
                height, width = gray.shape
                x = 5
                y = 5
                w = width - 10
                h = height - 10

                face = gray[y:y + h, x:x + w]
                face_resized = cv2.resize(face, (resized_width, resized_height))
                #cv2.equalizeHist(face_resized, face_resized)
                #name = fn + "converted"
                cv2.imwrite(path, face_resized)
                #print "width: " + str(width) + ", height: " + str(height)

    print "Conversion completed successfully"
    return

eigen_train_data()

