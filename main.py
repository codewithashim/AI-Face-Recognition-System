import cv2
import numpy as np
import os
import face_recognition
import pickle

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

imgBackround = cv2.imread('Resources/background.png')

# importing the mode images list from the folder
folderModePath = "Resources/Modes/"
mdoePathList = os.listdir(folderModePath)
imgModeList = []

for modePath in mdoePathList:
    imgModeList.append(cv2.imread(os.path.join(folderModePath, modePath)))

# Load EncodeFile
print("Loading Encode File ...")
file = open('EncodeFile', 'rb')
encodeListKnownWithIds = pickle.load(file)
file.close()
encodeListKnown, studentIds = encodeListKnownWithIds
# print(studentIds)
print("Encode File Loaded")

while True:
    success, img = cap.read()

    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
    
    faceCurFrame = face_recognition.face_locations(imgS)
    encodeCurFrame = face_recognition.face_encodings(imgS, faceCurFrame)

    imgBackround[162:162+480, 55:55+640] = img
    imgBackround[44:44+633, 808:808+414] = imgModeList[3]
    
    for encodeFace, faceLoc in zip(encodeCurFrame, faceCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        matchIndex = np.argmin(faceDis)
        
        if matches[matchIndex]:
            name = studentIds[matchIndex].upper()
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = int(y1*4.5), int(x2*4.5), int(y2*4.5), int(x1*4.5)
            cv2.rectangle(imgBackround, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(imgBackround, (x1, y2-35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(imgBackround, name, (x1+6, y2-6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
        else:
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = int(y1*4.5), int(x2*4.5), int(y2*4.5), int(x1*4.5)
            cv2.rectangle(imgBackround, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.rectangle(imgBackround, (x1, y2-35), (x2, y2), (0, 0, 255), cv2.FILLED)
            cv2.putText(imgBackround, "Unknown", (x1+6, y2-6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow("Face Attendance", imgBackround)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
