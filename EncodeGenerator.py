import cv2
import numpy as np
import os
import face_recognition
import pickle

# import the images list from the folder
folderPath = "Images"
pathList = os.listdir(folderPath)
imgList = []
studentIds = []

for path in pathList:
    imgList.append(cv2.imread(os.path.join(folderPath, path)))
    studentIds.append(os.path.splitext(path)[0])

def findEncoding(imgList):
    encodeList = []
    for img in imgList:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

encodeListKnown = findEncoding(imgList)
encodeListKnownWithIds = [encodeListKnown, studentIds]

print("Encoding Complete" ,encodeListKnownWithIds)

file = open('EncodeFile', 'wb')
pickle.dump(encodeListKnownWithIds, file)
file.close()