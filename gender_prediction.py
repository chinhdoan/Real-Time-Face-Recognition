import cv2
from mtcnn.mtcnn import MTCNN


frame = cv2.VideoCapture(0)

detector= MTCNN()

ageProto="./data/age_deploy.prototxt"
ageModel="./data/age_net.caffemodel"
genderProto="./data/gender_deploy.prototxt"
genderModel="./data/gender_net.caffemodel"

MODEL_MEAN_VALUES=(78.4263377603, 87.7689143744, 114.895847746)
ageList=['(0-2)', '(4-6)', '(8-12)', '(15-20)','(21-24)' ,'(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList=['Male','Female']

face_cascade = cv2.CascadeClassifier('./data/haarcascade_frontalface_default.xml')
ageNet=cv2.dnn.readNet(ageModel,ageProto)
genderNet=cv2.dnn.readNet(genderModel,genderProto)


def ageAndgender():

    while True:
        ret, img = frame.read()
        default_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        face = face_cascade.detectMultiScale(image=default_img, scaleFactor=1.3, minNeighbors=5)
        for x, y, w, h in face:
            roi = default_img[y:y + h, x:x + w]
            blob = cv2.dnn.blobFromImage(roi, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
            genderNet.setInput(blob)
            genderPreds = genderNet.forward()
            gender = genderList[genderPreds[0].argmax()]
            ageNet.setInput(blob)
            agePreds = ageNet.forward()
            age = ageList[agePreds[0].argmax()]
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(img, f"{gender}, {age} year", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255))

        cv2.imshow("Gender and Age Prediction", img)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:
                break
    cv2.destroyAllWindows()