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
        "major": "Software Engineer",
        "starting_year": 2019,
        "total_attendance": 2,
        "standing": "G",
        "year": 4,
        "last_attendance_time": "2022-06-11 05:07:00"
    },
    "852741":
    {
        "name": "Emly Blunt",
        "major": "Economics",
        "starting_year": 2021,
        "total_attendance": 12,
        "standing": "B",
        "year": 1,
        "last_attendance_time": "2022-12-11 00:54:34"
    },
    "963852":
        {
            "name": "Elon Musk",
            "major": "Physics",
            "starting_year": 2020,
            "total_attendance": 7,
            "standing": "G",
            "year": 2,
            "last_attendance_time": "2022-12-11 00:54:34"
    }
}

for key, value in data.items():
    ref.child(key).set(value)
