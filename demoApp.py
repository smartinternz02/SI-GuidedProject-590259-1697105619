import cv2



# Load weight file
weight = r"YOLO-V4/yolov4-custom_last.weights"
# Load config file
config = r"YOLO-V4/yolov4-custom.cfg"
# Initialize class names
classes = ["Handgun", "Knife"]
# Loading yolo with opencv
net = cv2.dnn.readNet(weight, config)
model = cv2.dnn.DetectionModel(net)
model.setInputParams(scale=1/255, size=(416, 416))

capp = cv2.VideoCapture(0)
# weapon = False
while True:
    # Capturing frames from video
    _, frame = capp.read()
    print(frame.shape)
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

    # Terminate the video
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
capp.release()
cv2.destroyAllWindows()

