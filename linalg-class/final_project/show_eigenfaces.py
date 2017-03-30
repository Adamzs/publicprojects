import numpy as np
import cv2
import sys
import os

def eigen_train_data():
    model = cv2.createEigenFaceRecognizer()
    model.load('eigen_trained_data_all.yml')
    evals = model.getMat('eigenvalues')
    (h, w) = evals.shape
    print "height=" + str(h) + ", width=" + str(w)
    total_all = 0
    for i in range(0, h):
        total_all += evals[i]
    print "total_all=" + str(total_all)

    model.load('eigen_trained_data_5.yml')
    evals = model.getMat('eigenvalues')
    (h, w) = evals.shape
    total = 0
    for i in range(0, h):
        total += evals[i]
    print "total_5=" + str(total) + ", " + str(total / total_all) + "%"

    model.load('eigen_trained_data_10.yml')
    evals = model.getMat('eigenvalues')
    (h, w) = evals.shape
    total = 0
    for i in range(0, h):
        total += evals[i]
    print "total_10=" + str(total) + ", " + str(total / total_all) + "%"

    model.load('eigen_trained_data_20.yml')
    evals = model.getMat('eigenvalues')
    (h, w) = evals.shape
    total = 0
    for i in range(0, h):
        total += evals[i]
    print "total_20=" + str(total) + ", " + str(total / total_all) + "%"

    model.load('eigen_trained_data_30.yml')
    evals = model.getMat('eigenvalues')
    (h, w) = evals.shape
    total = 0
    for i in range(0, h):
        total += evals[i]
    print "total_30=" + str(total) + ", " + str(total / total_all) + "%"

    model.load('eigen_trained_data_40.yml')
    evals = model.getMat('eigenvalues')
    (h, w) = evals.shape
    total = 0
    for i in range(0, h):
        total += evals[i]
    print "total_40=" + str(total) + ", " + str(total / total_all) + "%"

    model.load('eigen_trained_data_50.yml')
    evals = model.getMat('eigenvalues')
    (h, w) = evals.shape
    total = 0
    for i in range(0, h):
        total += evals[i]
    print "total_50=" + str(total) + ", " + str(total / total_all) + "%"

    model.load('eigen_trained_data_60.yml')
    evals = model.getMat('eigenvalues')
    (h, w) = evals.shape
    total = 0
    for i in range(0, h):
        total += evals[i]
    print "total_60=" + str(total) + ", " + str(total / total_all) + "%"

    model.load('eigen_trained_data_70.yml')
    evals = model.getMat('eigenvalues')
    (h, w) = evals.shape
    total = 0
    for i in range(0, h):
        total += evals[i]
    print "total_70=" + str(total) + ", " + str(total / total_all) + "%"

    model.load('eigen_trained_data_80.yml')
    evals = model.getMat('eigenvalues')
    (h, w) = evals.shape
    total = 0
    for i in range(0, h):
        total += evals[i]
    print "total_80=" + str(total) + ", " + str(total / total_all) + "%"


#    W = model.getMat('eigenvectors')
#    (r, c) = W.shape
#    for i in range(0, c):
#        v1 = W[:,[i]]
#        eimage = np.reshape(v1, (112, 92))
#        cv2.normalize(eimage, eimage, 0, 255, cv2.NORM_MINMAX)
#        cv2.imwrite('e%s.png' % i, eimage)

#    print "Training completed successfully" + str(r) + "," + str(c)


    return

eigen_train_data()

