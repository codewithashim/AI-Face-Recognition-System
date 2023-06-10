import firebase_admin
from firebase_admin import credentials
from firebase_admin import db

cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': "https://ai-face-recognition-system-default-rtdb.firebaseio.com/"
})

ref = db.reference('Students')

data = {
    "493549": {
        "name": "Ashim Rudra Paul",
        "mejor": "Software Engineering",
        "startingYear": "2019",
        "totalAttendance": 1,
        "totalClass": 0,
        "year": 4,
        "last_attendance": "2022-06-11 05:07:00",
    }
}

for key,value in data.items():
    ref.child(key).set(value)