import cv2
from mtcnn.mtcnn import MTCNN
from keras.models import load_model
from keras.preprocessing.image import img_to_array
import numpy as np


frame = cv2.VideoCapture(0)

detector= MTCNN()

emotion_model = "./data/_mini_XCEPTION.106-0.65.hdf5"


MODEL_MEAN_VALUES=(78.4263377603, 87.7689143744, 114.895847746)
Emotions = ["angry","disgust","scared","happy","sad","surprised","neutral"]

face_cascade = cv2.CascadeClassifier('./data/haarcascade_frontalface_default.xml')
emotion_classifier = load_model(emotion_model,compile=False)



def emotion():

    while True:
        ret, img = frame.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face = face_cascade.detectMultiScale(image=gray, scaleFactor=1.3, minNeighbors=5)

        for x, y, w, h in face:

            roi = gray[y:y + h, x:x + w]
            roi = cv2.resize(roi, (48, 48))
            roi = roi.astype("float") / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)
            preds = emotion_classifier.predict(roi)[0]
            emotion_probability = np.max(preds)
            label = Emotions[preds.argmax()]
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(img, f"{label}", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0))

        cv2.imshow("Gender and Age Prediction", img)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:
                break
    cv2.destroyAllWindows()