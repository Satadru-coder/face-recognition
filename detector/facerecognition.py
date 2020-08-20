import cv2
import sys
import pickle

count=0

cascPath= "haarcascade_frontalface_default.xml"
faceCascade= cv2.CascadeClassifier(cascPath)

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer.yml")

labels={}
with open("labels.pickel","rb") as f:
    bslabels=pickle.load(f)
    labels= {v:k for k,v in bslabels.items()}

video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30,30),
        flags=cv2.FONT_HERSHEY_SIMPLEX
    )

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        roi_gray= gray[y:y+h, x:x+w]

        id_, conf = recognizer.predict(roi_gray)
        if conf<35:
            #print(id_)
            #print(labels[id_])
            font= cv2.FONT_HERSHEY_SIMPLEX
            name = labels[id_]
            color= [255, 255, 255]
            cv2.putText(frame,name,(x,y),font,1,color,2, cv2.LINE_AA)

        cv2.imshow('Video', frame)
        print(id_,conf)
        #print(name)
        #cv2.imwrite("sex.png", roi_gray)
        #count+=1

    if cv2.waitKey(1) & 0xFF == ord(' '):
        break
video_capture.release()
cv2.destroyAllWindows()
