import numpy as np
import cv2
import sys
import os

RESIZE_FACTOR = 4

def eigen_train_data():
    face_dir = 'face_data'
    resized_width, resized_height = (92, 112)
    cascadeFile = "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascadeFile)

    for (subdirs, dirs, files) in os.walk(face_dir):
        for subdir in dirs:
            img_path = os.path.join(face_dir, subdir)
            for fn in os.listdir(img_path):
                path = img_path + '/' + fn
                gray = cv2.imread(path, 0)
                #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(30, 30),
                    flags=cv2.cv.CV_HAAR_SCALE_IMAGE
                )
                if len(faces) > 0:
                    areas = []
                    for (x, y, w, h) in faces:
                        areas.append(w * h)
                    max_area, idx = max([(val, idx) for idx, val in enumerate(areas)])
                    face_sel = faces[idx]

                    x = max((face_sel[0] * RESIZE_FACTOR), 0)
                    y = max((face_sel[1] * RESIZE_FACTOR) - 10, 0)
                    w = min((face_sel[2] * RESIZE_FACTOR), gray.shape[1])
                    h = min((face_sel[3] * RESIZE_FACTOR) + 25, gray.shape[0])

                    face = gray[y:y + h, x:x + w]
                    if len(face) > 0:
                        face_resized = cv2.resize(face, (resized_width, resized_height))
                        # cv2.equalizeHist(face_resized, face_resized)
                        name = fn + "converted"
                        cv2.imwrite('%s/%s.png' % (img_path, name), face_resized)

    print "Conversion completed successfully"
    return

eigen_train_data()

