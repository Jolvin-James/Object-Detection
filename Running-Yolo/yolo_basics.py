from ultralytics import YOLO
import cv2

model = YOLO('../Yolo-Weights/yolov8l.pt') # specify the weights like n -> nano l -> large
results = model("Running Yolo/1.png", show=True) 
cv2.waitKey(0) # unless the user doesn't quit keep displaying

