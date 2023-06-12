import cv2
import numpy as np
import os
import face_recognition
import pickle
import cvzone
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
from firebase_admin import storage
import numpy as np
from datetime import datetime

cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': "https://ai-face-recognition-system-default-rtdb.firebaseio.com/",
    'storageBucket': "gs://ai-face-recognition-system.appspot.com"
})

bucket = storage.bucket()

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

imgBackground = cv2.imread('Resources/background.png')

# importing the mode images list from the folder
folderModePath = "Resources/Modes/"
modePathList = os.listdir(folderModePath)
imgModeList = []

for modePath in modePathList:
    imgModeList.append(cv2.imread(os.path.join(folderModePath, modePath)))

# Load EncodeFile
print("Loading Encode File ...")
file = open('EncodeFile.p', 'rb')
encodeListKnownWithIds = pickle.load(file)
file.close()
encodeListKnown, studentIds = encodeListKnownWithIds
print("Encode File Loaded")

modeType = 0
counter = 0
id = -1
imgStudent = []

try:
    while True:
        success, img = cap.read()

        imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
        imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

        faceCurFrame = face_recognition.face_locations(imgS)
        encodeCurFrame = face_recognition.face_encodings(imgS, faceCurFrame)

        imgBackground[162:162 + 480, 55:55 + 640] = img
        imgBackground[44:44 + 633, 808:808 + 414] = imgModeList[modeType]

        for encodeFace, faceLoc in zip(encodeCurFrame, faceCurFrame):
            matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
            faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)

            matchIndex = np.argmin(faceDis)

            if matches[matchIndex]:
                y1, x2, y2, x1 = faceLoc
                y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                bbox = 55 + x1, 162 + y1, x2 - x1, y2 - y1
                imgBackground = cvzone.cornerRect(imgBackground, bbox, rt=0)
                id = studentIds[matchIndex]
                if counter == 0:
                    cvzone.putTextRect(imgBackground, "Loading", (275, 400))
                    cv2.imshow("Face Attendance", imgBackground)
                    cv2.waitKey(1)
                    counter = 1
                    modeType = 1
            if counter != 0:

                if counter == 1:
                    # Get the Data
                    studentInfo = db.reference(f'Students/{id}').get()
                    print(studentInfo)
                    # Get the Image from the storage
                    blob = bucket.get_blob(f'Images/{id}.png')
                    if blob is not None:
                        array = np.frombuffer(blob.download_as_string(), np.uint8)
                        imgStudent = cv2.imdecode(array, cv2.IMREAD_COLOR)
                        imgStudent = cv2.resize(imgStudent, (216, 216))
                        counter = 2
                elif counter == 2:
                    # Add Student Image to Background
                    imgBackground[175:175 + 216, 909:909 + 216] = imgStudent
                    counter = 3
                elif counter == 3:
                    # Add Text to the Image
                    cv2.putText(imgBackground, studentInfo['name'], (861, 95),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                    cv2.putText(imgBackground, str(studentInfo['total_attendance']), (861, 125),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                    date_string = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
                    cv2.putText(imgBackground, date_string, (861, 210),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
                    counter = 4
                elif counter == 4:
                    # Save Attendance
                    today = datetime.now().strftime("%Y-%m-%d")
                    attendanceRef = db.reference(f'Students/{id}/attendance/{today}')
                    attendanceRef.set(True)
                    counter = 5

        cv2.imshow("Face Attendance", imgBackground)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    pass

cap.release()
cv2.destroyAllWindows()
