import os
from PIL import Image
import numpy as np
import cv2
import pickle

basedir= os.path.dirname(os.path.abspath(__file__))
imagedir= os.path.join(basedir,"images")

cascPath= "haarcascade_frontalface_default.xml"
faceCascade= cv2.CascadeClassifier(cascPath)



recognizer = cv2.face.LBPHFaceRecognizer_create()

currentid=0
label_id={}
xtrain=[]
ylabels=[]

for root, dirs, files in os.walk(imagedir):
    for file in files:
        if file.endswith("png") or file.endswith("jpg"):
            path = os.path.join(root, file)
            label= os.path.basename(os.path.dirname(path)).replace(" ","_").lower()
            #print(label,path)
            if not label in label_id:
                label_id[label]=currentid
                currentid += 1
            Id= label_id[label]
            #print(label_id)


            pilimage= Image.open(path).convert("L") #convert grayscale
            imag= pilimage.resize((100,100), Image.ANTIALIAS)
            imgarray= np.array(pilimage,"uint8") #converting a image into a numpy array
            #print(imgarray)
            faces= faceCascade.detectMultiScale(imgarray, scaleFactor=1.5, minNeighbors=5, minSize=(30,30), flags=cv2.FONT_HERSHEY_SIMPLEX)

            for (x, y, w, h) in faces:
                roi= imgarray[y:y+h, x:x+w]
                xtrain.append(roi)
                ylabels.append(Id)

with open("labels.pickel","wb") as f:
    pickle.dump(label_id,f)

recognizer.train(xtrain,np.array(ylabels))
recognizer.save("trainer.yml")