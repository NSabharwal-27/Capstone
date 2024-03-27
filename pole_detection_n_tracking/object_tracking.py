import cv2
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture(0)

ret = True

while ret:
    ret, frame = cap.read()

    results = model.track(frame, persist=True)

    frame_ = results[0].plot()

    cv2.imshow('Frame', frame_)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break