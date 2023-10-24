import requests
import cv2 as cv
import numpy as np
import imutils
from ultralytics import YOLO

# Replace the below URL with your own. Make sure to add "/shot.jpg" at last.
url = "http://192.168.178.39:8080/shot.jpg"
model = YOLO('/home/moiki/Documents/Files/studies/4_Semester/ADL/ADL_W23/models/large_dataset_best_1.pt')


# While loop to continuously fetching data from the Url
while True:
    img_resp = requests.get(url) 
    img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8)
    img = cv.imdecode(img_arr, -1)
    img = imutils.resize(img, width=1000, height=1800)
    results = model(source = img, show = True, conf = 0.4)

    # Press Esc key to exit
    if cv.waitKey(1) == 27:
        break

cv.destroyAllWindows()