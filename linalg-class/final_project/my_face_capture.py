import numpy as np
import cv2
import sys
import os

FREQ_DIV = 5  # frequency divider for capturing training images
RESIZE_FACTOR = 4
NUM_TRAINING = 100

def capture_training_images():
    if len(sys.argv) < 2:
        return

    video_capture = cv2.VideoCapture(0)
    cascadeFile = "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascadeFile)

    face_dir = 'face_data'
    face_name = sys.argv[1]
    image_path = os.path.join(face_dir, face_name)
    if not os.path.isdir(image_path):
        os.mkdir(image_path)

    count_captures = 0
    count_timer = 0

    while True:
        count_timer += 1
        ret, frame = video_capture.read()
        inImg = np.array(frame)
        outImg, count_captures = process_image(inImg, face_cascade, face_name, image_path, count_captures, count_timer)
        cv2.imshow('Video', outImg)

        # When everything is done, release the capture on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            video_capture.release()
            cv2.destroyAllWindows()
            return

def process_image(inImg, face_cascade, face_name, file_path, count_captures, count_timer):
    frame = cv2.flip(inImg, 1)
    resized_width, resized_height = (92, 112)
    if count_captures < NUM_TRAINING:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_resized = cv2.resize(gray, (int(gray.shape[1] / RESIZE_FACTOR), int(gray.shape[0] / RESIZE_FACTOR)))
        faces = face_cascade.detectMultiScale(
            gray_resized,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
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
            face_resized = cv2.resize(face, (resized_width, resized_height))
            img_no = sorted([int(fn[:fn.find('.')]) for fn in os.listdir(file_path) if fn[0] != '.'] + [0])[-1] + 1
            #cv2.equalizeHist(face_resized, face_resized)

            if count_timer % FREQ_DIV == 0:
                cv2.imwrite('%s/%s.png' % (file_path, img_no), face_resized)
                count_captures += 1
                print("Captured image: " + str(count_captures) + ", (" + str(gray.shape[1]) + "," + str(
                    gray.shape[0]) + ")")

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
            cv2.putText(frame, face_name, (x - 10, y - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))
    elif count_captures == NUM_TRAINING:
        print ("Training data captured. Press 'q' to exit.")
        count_captures += 1

    return (frame, count_captures)

capture_training_images()
