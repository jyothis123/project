import numpy as np
import argparse
import matplotlib.pyplot as plt
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
import os
#import pyttsx3
import time
#from recog_face import predict
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# command line argument

model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(612, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(5, activation='softmax'))
mode='train' 
def start(frame):
        n_prev=''
        model.load_weights('model.h5')
    # prevents openCL usage and unnecessary logging messages
        cv2.ocl.setUseOpenCL(False)
    # dictionary which assigns each label an emotion (alphabetical order)
        emotion_dict = {0: "Angry", 1: "Happy", 2: "Neutral", 3: "Sad", 4: "Surprised"}
    # start the webcam feed
    #cap = cv2.VideoCapture(0)
   # while True:
        #Find haar cascade to draw bounding box around face
       # ret,frame = cap.read()
        
        facecasc = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        print(facecasc)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = facecasc.detectMultiScale(gray,scaleFactor=1.3, minNeighbors=5)
        labels=[]
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)
            roi_gray = gray[y:y + h, x:x + w]
            cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
            prediction = model.predict(cropped_img)
            print(prediction)
            maxindex = int(np.argmax(prediction))
            labels.append(emotion_dict[maxindex])
            print(labels)
            cv2.putText(frame, emotion_dict[maxindex], (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            break
        cv2.imshow('Video', cv2.resize(frame,(1600,960),interpolation = cv2.INTER_CUBIC))
        #if cv2.waitKey(1) & 0xFF == ord('q'):
         #   break
    
        #cap.release()
        #cv2.destroyAllWindows()
        cv2.waitKey(1)
   
        return labels
def speak1(text): 
 for i in text:
       print("texttt",i)
       #engine = pyttsx3.init()
       #engine.say(i)
       #engine.runAndWait()
       print("done")
       time.sleep(1)    
def main1():  
 cap = cv2.VideoCapture(0)
 i = 0
 
 while(cap.isOpened()):
    ret, frame = cap.read()
    # cv2.imshow("Image", frame)
    # cv2.waitKey(1)
    label=start(frame)
   
    speak1(label)
    if cv2.waitKey(1) & 0xFF == ord('q'):
            break

 cap.release()
 cv2.destroyAllWindows()

main1()
