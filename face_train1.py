from imutils import paths
import face_recognition
import argparse
import pickle
import cv2
import os
from tkinter import messagebox
import sys
import numpy as np

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
        if confidence > 0.7:
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




def training():
        # grab the paths to the input images in our dataset
        print("[INFO] quantifying faces...")
        imagePaths = list(paths.list_images("face_dataset"))

        # initialize the list of known encodings and known names
        knownEncodings = []
        knownNames = []

        # loop over the image paths
        for (i, imagePath) in enumerate(imagePaths):
                # extract the person name from the image path
                print("[INFO] processing image {}/{}".format(i + 1,
                        len(imagePaths)))
                name = imagePath.split(os.path.sep)[-2]
                print(name)
               

                # load the input image and convert it from RGB (OpenCV ordering)
                # to dlib ordering (RGB)
                image = cv2.imread(imagePath)
                print(image.shape)
                # sys.exit()

                #image=cv2.resize(image,(300,300))
                rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                # detect the (x, y)-coordinates of the bounding boxes
                # corresponding to each face in the input image
                # boxes = face_recognition.face_locations(rgb,
                #         model="cnn")
                boxes = detect_face(rgb,faceNet)

                # compute the facial embedding for the face
                encodings = face_recognition.face_encodings(rgb, boxes)

                # loop over the encodings
                for encoding in encodings:
                        # add each encoding + name to our set of known names and
                        # encodings
                        knownEncodings.append(encoding)
                        knownNames.append(name)

        # dump the facial encodings + names to disk
        print("[INFO] serializing encodings...")
        data = {"encodings": knownEncodings, "names": knownNames}
        f = open("encodings.pickle", "wb")
        f.write(pickle.dumps(data))
        f.close()
        print("saved to disk")
training()