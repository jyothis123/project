from tkinter import *
import cv2
import numpy as np
import pyttsx3
import time
import face_recognition
import argparse
import imutils
import pickle
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from keras.models import load_model
import os
import sys
import paho.mqtt.client as mqtt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

#######################model of emotion-detection##############################
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#face detection################################################
encodings="encodings.pickle"
detection_method="cnn"
display=1
# load the known faces and embeddings
print("[INFO] loading encodings...")
data = pickle.loads(open(encodings, "rb").read())
from PIL import Image, ImageTk

def detect():
    top = Tk() 
    top.title("gui")
    top.geometry("1000x700")
    e=Frame(top)
    e.config(bg="lightblue")
    e.pack(fill=BOTH, expand=YES)
    try:
        path3="eye.png"
        image = Image.open(path3)
    except:
        path3="eye.png"
        image = Image.open(path3)
        messagebox.showerror("Not an image","Choose an image") 
    photo = ImageTk.PhotoImage(image.resize((1000, 700), Image.ANTIALIAS))
    label = Label(top, image=photo, bg='yellow')
    label.image = photo
    label.pack()
    lbl1=Label(top,text="Emotion,Face and Object detection for blind",font ="Helvetica 30 bold ")
    lbl1.place(x=10,y=100)
    lb2=Label(top,text="Please click Button for Face Recognition",font ="Helvetica 10 bold ")
    lb2.place(x=300,y=275)
    btn = Button(top, text="check",bg="sandy brown",font="25px",command=lambda:start())
    btn.place(x=600, y=275)
    lb2=Label(top,text="Please click Button for Emotion Detection",font ="Helvetica 10 bold ")
    lb2.place(x=300,y=325)
    btn = Button(top, text="check",bg="sandy brown",font="25px",command=lambda:main1())
    btn.place(x=600, y=325)
    lb2=Label(top,text="Please click Button for Object Detection",font ="Helvetica 10 bold ")
    lb2.place(x=300,y=380)
    global btn_test
    btn_test = Button(top, text="check",bg="sandy brown",font="25px",command=lambda:start1())
    btn_test.place(x=600, y=380)

    #btn = Button(top, text="stop",bg="sandy brown",font="25px",command=lambda:start1())
    #btn.place(x=420, y=380)
    top.mainloop()
 #emotion detection:main############################################################3
def start2(frame):
        n_prev=''
        model=load_model('model_5.hdf5')
    # prevents openCL usage and unnecessary logging messages
        cv2.ocl.setUseOpenCL(False)
    # dictionary which assigns each label an emotion (alphabetical order)
        emotion_dict = {0: "Angry", 1: "Sad", 2: "Sad", 3: "Happy", 4: "Sad",5:'Surprise',6:'neutral'}
    # s
        
        facecasc = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = facecasc.detectMultiScale(gray,scaleFactor=1.3, minNeighbors=5)
        labels=[]
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)
            cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)
            face_crop = frame[y:y + h, x:x + w]
            face_crop = cv2.resize(face_crop, (48, 48))
            face_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
            face_crop = face_crop.astype('float32') / 255
            face_crop = np.asarray(face_crop)
            face_crop = face_crop.reshape(1, 1, face_crop.shape[0], face_crop.shape[1])
            prediction = model.predict(face_crop)
            maxindex = int(np.argmax(prediction))
            labels.append(emotion_dict[maxindex])
            cv2.putText(frame, emotion_dict[maxindex], (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            break
        cv2.imshow('Video', cv2.resize(frame,(700,700),interpolation = cv2.INTER_CUBIC))
        #if cv2.waitKey(1) & 0xFF == ord('q'):
         #   break
    
        #cap.release()
        #cv2.destroyAllWindows()
        cv2.waitKey(1)
   
        return labels
########################################################################################################

#====================face detection
prototxtPath ="deploy.prototxt.txt"
weightsPath ="res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)
def detect_face(frame,faceNet):
    # grab the dimensions of the frame and then construct a blob
    # from it
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame,(300,300)), 1.0, (300, 300),
        (104.0, 177.0, 123.0))
    # pass the blob through the network and obtain the face detections
    faceNet.setInput(blob)
    detections = faceNet.forward()
    # initialize our list of faces, their corresponding locations,
    # and the list of predictions from our face mask network
    faces = []
    locs = []
    preds = []
    # loop over the detections
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        #print(confidence)
        if confidence > 0.8:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            # (startX, startY, endX, endY)=(startX, startY-50, endX, endY+50)
            # (startX, startY) = (max(0, startX), max(0, startY))
            # (endX, endY) = (min(w - 1, endX), min(h - 1, endY))
            
            locs.append((int(startY), int(endX), int(endY), int(startX)))  
            cv2.rectangle(frame, (int(startX),  int(startY)), (int(endX),int(endY)), (0, 0, 255), 2)   
    # only make a predictions if at least one face was detected
    
    # return a 2-tuple of the face locations and their corresponding
    # locations
    cv2.imwrite('re.png',frame)
    return locs   
#######################################################################     

#yolo##############################################################
net = cv2.dnn.readNet("Project_Extra/yolov4.weights", "Project_Extra/yolov4.cfg")
classes = []
with open("Project_Extra/labels.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers =[layer_names[i[0]-1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))
##########################################################
def detect_person(yimg):
    yimg = cv2.resize(yimg, None, fx=0.7, fy=0.7)
    height, width, channels = yimg.shape

    # Detecting objects
    blob = cv2.dnn.blobFromImage(yimg, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

    net.setInput(blob)
    outs = net.forward(output_layers)
    
    # Showing informations on the screenb
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            
            scores = detection[5:]
            class_id = np.argmax(scores)
            
            confidence = scores[class_id]
            
            if confidence > 0.5:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    font = cv2.FONT_HERSHEY_PLAIN
    labels=[]
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            labels.append(label)
            color = colors[i]
            cv2.rectangle(yimg, (x-10, y-10), (x + w+20, y + h+20), color, 2)
            # crop_img = img[y-10:y + h+20, x-10:x + w+20]
            # cv2.imshow("cropped", crop_img)
            # cv2.waitKey(0)
            cv2.putText(yimg, label, (x, y-15), font, 1, color, 2)
           


    cv2.imshow("Image", yimg)
    cv2.waitKey(1)
    return labels
#############################face-reco:prediction#################################
def predict(frame):
    
    # frame=cv2.imread(path)
    # frame=cv2.resize(frame,(300,300))
    
    
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # rgb = imutils.resize(frame, width=300)
    r = frame.shape[1] / float(rgb.shape[1])
    r=1

    
    boxes1 = detect_face(rgb,faceNet)

    # print(type(boxes[0]))
    print(boxes1)
    print('====')
    # s
    encodings = face_recognition.face_encodings(rgb, boxes1)
    names = []
    
    # loop over the facial embeddings
    for encoding in encodings:
            print(encoding)
            # attempt to match each face in the input image to our known
            # encodings
            matches = face_recognition.compare_faces(data["encodings"],
                    encoding)
            name = "unknown"

            # check to see if we have found a match
            if True in matches:
                    # find the indexes of all matched faces then initialize a
                    # dictionary to count the total number of times each face
                    # was matched
                    matchedIdxs = [i for (i, b) in enumerate(matches) if b]
                    counts = {}

                    # loop over the matched indexes and maintain a count for
                    # each recognized face face
                    for i in matchedIdxs:
                            name = data["names"][i]
                            counts[name] = counts.get(name, 0) + 1

                    # determine the recognized face with the largest number
                    # of votes (note: in the event of an unlikely tie Python
                    # will select first entry in the dictionary)
                    name = max(counts, key=counts.get)
            
                    # update the list of names
            names.append(name)
            print(names)
    # loop over the recognized faces
    
    for ((top, right, bottom, left), name) in zip(boxes1, names):
            # rescale the face coordinates
            try:
                fcc=frame[top:top+bottom,left:left+right]
                img=cv2.resize(fcc,(224,224))
                img=img_to_array(img)
                img=preprocess_input(img)
                img=img.reshape((1,224,224,3))
                color=(0,0,255)
                
                top = int(top * r)
                right = int(right * r)
                bottom = int(bottom * r)
                left = int(left * r)

                # draw the predicted face name on the image
                cv2.rectangle(frame, (left, top), (right, bottom),
                        color, 2)
                y = top - 15 if top - 15 > 15 else top + 15
                cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.75, color, 2)
                cv2.imshow("frame",frame)
                cv2.waitKey(1)
                print(name)        
            except Exception as e:
                print(e)
                continue        

    if(len(names)==0):
        return "no face"
    else:
        return name
#################################object detection-end####################################
def speak(text):
 res=[]
 for i in text:
    if i not in res:
      res.append(i)
      for j in res:
       print("texttt",res)
      
       engine = pyttsx3.init()
       engine.say(j)
       engine.runAndWait()
       print("done")
       time.sleep(1)
#def stop():
 #pass
####################object detection-start#############################################
def start1():
 cap = cv2.VideoCapture(0)
 i = 0
 
 while(cap.isOpened()):
    ret, frame = cap.read()
    # cv2.imshow("Image", frame)
    # cv2.waitKey(1)
    label=detect_person(frame)
   
    speak(label)
    if cv2.waitKey(1) & 0xFF == ord('q'):
            break

 cap.release()
 cv2.destroyAllWindows()
###############################face detection-end###############################################
def speak1(text):
 
       print("texttt",text)
       
       engine = pyttsx3.init()
       engine.say(text)
       engine.runAndWait()
       print("done")
       time.sleep(1)

print("[INFO] processing video...")
#################face detection-start#######################################################
def start():
 stream = cv2.VideoCapture(0)
 writer = None
# loop over frames from the video file stream
 while True:
        # grab the next frame
        (grabbed, frame) = stream.read()
        
        res=predict(frame)
        speak1(res)
#####################emotion detection-end################################################
def speak2(text): 
 for i in text:
       print("texttt",i)
       engine = pyttsx3.init()
       engine.say(i)
       engine.runAndWait()
       print("done")
       time.sleep(1) 
#####################emotion detection-start############################################## 
def main1():  
 cap = cv2.VideoCapture(0)
 i = 0
 
 while(cap.isOpened()):
    ret, frame = cap.read()
    # cv2.imshow("Image", frame)
    # cv2.waitKey(1)
    label=start2(frame)
   
    speak2(label)
    if cv2.waitKey(1) & 0xFF == ord('q'):
            break

 cap.release()
 cv2.destroyAllWindows()
##################################################################################################
if __name__=="__main__":
    detect()
    

    