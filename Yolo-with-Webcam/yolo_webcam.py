from ultralytics import YOLO
import cv2
import cvzone 
import math

# for webcam
# cap = cv2.VideoCapture(0) # if you have only 1 webcam then 0 and if multiple then 1
# cap.set(3, 1280)  # setting the width and height
# cap.set(4, 720)

# for video capturing
cap = cv2.VideoCapture("../Object Detection/Yolo with Webcam/bike.mp4")
# Check if the video capture was successful
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()


model = YOLO("../Yolo-Weights/yolov8n.pt")

# this is the coco dataset
classNames = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus",
    "train", "truck", "boat", "traffic light", "fire hydrant",
    "stop sign", "parking meter",
    "bench", "bird", "cat", "dog", "horse",
    "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
    "backpack", "umbrella", "handbag", "tie","suitcase","frisbee",
    "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife",
    "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", 
    "chair", "couch", "potted plant", "bed", "dining table",
    "toilet", "tv", "laptop",
    "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", 
    "vase", "scissors", "teddy bear", "hair dryer", "toothbrush"
]


while True:
    success, img = cap.read()
    results = model(img, stream=True)
    # for looping through and creating the boundary boxes
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3) # second last is for color and then its width of box
            
            w, h = x2-x1, y2-y1
            cvzone.cornerRect(img, (x1, y1, w, h))

            # Confidence
            conf = math.ceil((box.conf[0]*100))/100 # confidence score

            # Class Name
            cls = int(box.cls[0])

            cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (max(0, x1), max(35, y1)), scale=1, thickness=1)

    cv2.imshow("Image", img)
    cv2.waitKey(1) # waiting for a second