from flask import Flask, render_template, request, session
import cv2
import os
from playsound import playsound
import ibm_db
import re

app = Flask(__name__)
app.secret_key = 'a'
conn = ibm_db.connect("DATABASE=bludb;HOSTNAME=2d46b6b4-cbf6-40eb-bbce-6251e6ba0300.bs2io90l08kqb1od8lcg.databases.appdomain.cloud;PORT=32328;SECURITY=SSL;SSLServerCertificate=DigiCertGlobalRootCA.crt;UID=ghk16229;PWD=7LChql7JQburybLm;", "", "")
print("connected")

# Load weight file
weight = r"YOLO-V4/yolov4-custom_last.weights"
# Load config file
config = r"YOLO-V4/yolov4-custom.cfg"
# Initialize class names
classes = ["Handgun", "Knife"]
# Loading yolo with opencv
net = cv2.dnn.readNet(weight, config)
model = cv2.dnn_DetectionModel(net)
model.setInputParams(scale=1/255, size=(416, 416))


@app.route('/')
def project():
    return render_template('index.html')


@app.route('/hero')
def home():
    return render_template('index.html')


@app.route('/home')
def home1():
    return render_template('after_login.html')


@app.route('/login')
def login():
    return render_template('login.html')


@app.route("/log", methods=['POST', 'GET'])
def login1():
    if request.method == "POST":
        email = request.form["email"]
        password = request.form["password"]
        sql = "SELECT * FROM REGISTER_WEAPON WHERE EMAIL=? AND PASSWORD=?"  # from db2 sql table
        stmt = ibm_db.prepare(conn, sql)
        # this username & password is should be same as db-2 details & order also
        ibm_db.bind_param(stmt, 1, email)
        ibm_db.bind_param(stmt, 2, password)
        ibm_db.execute(stmt)
        account = ibm_db.fetch_assoc(stmt)
        print(account)
        if account:
            session['Loggedin'] = True
            session['id'] = account['EMAIL']
            session['email'] = account['EMAIL']
            return render_template('after_login.html')
        else:
            msg = "Incorrect Email/password"
            return render_template('login.html', msg=msg)
    else:
        return render_template('login.html')


@app.route('/register')
def register():
    return render_template('register.html')


@app.route("/reg", methods=['POST', 'GET'])
def signup():
    msg = ''
    if request.method == 'POST':
        name = request.form["name"]
        email = request.form["email"]
        password = request.form["password"]
        sql = "SELECT * FROM REGISTER_WEAPON WHERE name= ?"
        stmt = ibm_db.prepare(conn, sql)
        ibm_db.bind_param(stmt, 1, name)
        ibm_db.execute(stmt)
        account = ibm_db.fetch_assoc(stmt)
        print(account)
        if account:
            return render_template('login.html', error=True)
        elif not re.match(r'[^@]+@[^@]+\.[^@]+', email):
            msg = "Invalid Email Address!"
        else:
            insert_sql = "INSERT INTO REGISTER_WEAPON VALUES (?,?,?)"
            prep_stmt = ibm_db.prepare(conn, insert_sql)
            # this username & password is should be same as db-2 details & order also
            ibm_db.bind_param(prep_stmt, 1, name)
            ibm_db.bind_param(prep_stmt, 2, email)
            ibm_db.bind_param(prep_stmt, 3, password)
            ibm_db.execute(prep_stmt)
            msg = "You have successfully registered !"
    return render_template('login.html', msg=msg)


@app.route('/image')
def image():
    return render_template('image.html')


@app.route('/predict', methods=["GET", "POST"])
def img_pred():
    if request.method == 'POST':
        f = request.files['file']  # requesting the file
        basepath = os.path.dirname('__file__')  # storing the file directory
        filepath = os.path.join(basepath, "uploads", f.filename)  # storing the file in uploads folder
        f.save(filepath)  # saving the file

        img = cv2.imread(filepath)  # load and reshaping the image
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Initialize variables for output of pre-trained model
        classID, scores, bboxes = model.detect(img, nmsThreshold=0.4, confThreshold=0.3)
        for classID, scores, bboxes in zip(classID, scores, bboxes):
            x, y, w, h = bboxes
            if scores >= 0.7:
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1)
                cv2.putText(img, (classes[classID] + ' accuracy: ' + (str(round((scores * 100), 2))) + '%'),
                            (x, y - 5),
                            cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 1)
                cv2.putText(img, 'Weapon Detected', (20, 30), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)
                # playsound("alarm.mp3")
                # weapon = True
        cv2.imshow('output', img)
        cv2.waitKey(0)


@app.route('/video')
def video():
    return render_template('video.html')


@app.route('/predict_video', methods=["GET", "POST"])
def vid_pred():
    if request.method == 'POST':
        f = request.files['file']  # requesting the file
        basepath = os.path.dirname('__file__')  # storing the file directory
        filepath = os.path.join(basepath, "uploads", f.filename)  # storing the file in uploads folder
        print(filepath)
        f.save(filepath)  # saving the file

        # Input 2
        cap = cv2.VideoCapture(filepath)
        weapon = False
        while True:
            # Capturing frames from video
            _, frame = cap.read()
            # print(frame.shape)
            # Resizing frame
            frame = cv2.resize(frame, (640, 480))
            # Initialize variables for output of pre-trained model
            classID, scores, bboxes = model.detect(frame, nmsThreshold=0.4, confThreshold=0.3)
            for classID, scores, bboxes in zip(classID, scores, bboxes):
                x, y, w, h = bboxes
                if scores >= 0.7:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)
                    cv2.putText(frame, (classes[classID] + ' accuracy: ' + (str(round((scores * 100), 2))) + '%'), (x, y - 5), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 1)
                    cv2.putText(frame, 'Weapon Detected', (20, 30), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)
                    # playsound("alarm.mp3")
                    # weapon = True
            cv2.imshow('video', frame)
            if weapon==True:
                playsound("alarm.mp3")
            # Terminate the video
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()


@app.route('/live')
def live():
    return render_template('live.html')


@app.route('/predict_live')
def liv_pred():
    # Input 2
    capp = cv2.VideoCapture(0)
    weapon = False
    while True:
        # Capturing frames from video
        _, frame = capp.read()
        #print(frame.shape)
        # Resizing frame
        #frame = cv2.resize(frame, (640, 480))
        # Initialize variables for output of pre-trained model
        classID, scores, bboxes = model.detect(frame, nmsThreshold=0.4, confThreshold=0.3)
        for classID, scores, bboxes in zip(classID, scores, bboxes):
            x, y, w, h = bboxes
            if scores >= 0.7:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)
                cv2.putText(frame, (classes[classID] + ' accuracy: ' + (str(round((scores * 100), 2))) + '%'), (x, y - 5), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 1)
                cv2.putText(frame, 'Weapon Detected', (20, 30), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)
                # playsound("alarm.mp3")
                # weapon = True
        cv2.imshow('video', frame)
        if weapon==True:
            playsound("alarm.mp3")
        # Terminate the video
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    capp.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    app.run(debug=True)

"""# Input 2
cap = cv2.VideoCapture("input_2.mp4")"""