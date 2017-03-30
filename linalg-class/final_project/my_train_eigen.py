import numpy as np
import cv2
import sys
import os

def eigen_train_data():
    face_dir = 'face_data'

    imgs = []
    tags = []
    index = 0

    for (subdirs, dirs, files) in os.walk(face_dir):
        for subdir in dirs:
            img_path = os.path.join(face_dir, subdir)
            for fn in os.listdir(img_path):
                path = img_path + '/' + fn
                tag = index
                img = cv2.imread(path, 0)
                imgs.append(cv2.resize(img, (92, 112)))
                tags.append(int(tag))
            index += 1
    (imgs, tags) = [np.array(item) for item in [imgs, tags]]

    model = cv2.createEigenFaceRecognizer(5)
    model.train(imgs, tags)
    model.save('eigen_trained_data_5.yml')

    # model = cv2.createEigenFaceRecognizer(10)
    # model.train(imgs, tags)
    # model.save('eigen_trained_data_10.yml')
    #
    # model = cv2.createEigenFaceRecognizer(30)
    # model.train(imgs, tags)
    # model.save('eigen_trained_data_30.yml')
    #
    # model = cv2.createEigenFaceRecognizer(40)
    # model.train(imgs, tags)
    # model.save('eigen_trained_data_40.yml')
    #
    # model = cv2.createEigenFaceRecognizer(50)
    # model.train(imgs, tags)
    # model.save('eigen_trained_data_50.yml')
    #
    # model = cv2.createEigenFaceRecognizer(60)
    # model.train(imgs, tags)
    # model.save('eigen_trained_data_60.yml')

    model = cv2.createEigenFaceRecognizer(70)
    model.train(imgs, tags)
    model.save('eigen_trained_data_70.yml')

    model = cv2.createEigenFaceRecognizer(80)
    model.train(imgs, tags)
    model.save('eigen_trained_data_80.yml')

    print "Training completed successfully"


    return

eigen_train_data()

