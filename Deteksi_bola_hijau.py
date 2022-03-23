import cv2
import numpy as np
import imutils
import argparse
from collections import deque

ap = argparse.ArgumentParser()
ap.add_argument("-b", "--buffer", type=int, default=64,
	help="max buffer size")
args = vars(ap.parse_args())

#color range
color_low = (29, 86, 6)
color_up = (64, 255, 255)
pts = deque(maxlen=args["buffer"])

kamera = cv2.VideoCapture(0)

while True:
    ret, frame = kamera.read()
    frame = imutils.resize(frame, width=600)
    blur = cv2.GaussianBlur(frame, (11,11), 0)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    layer = cv2.inRange(hsv, color_low, color_up)
    layer = cv2.erode(layer, None, iterations = 3)
    layer = cv2.dilate(layer, None, iterations = 3)
    kontur = cv2.findContours(layer.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    kontur = imutils.grab_contours(kontur)
    pusat = None

    if len(kontur)>0:
        c = max(kontur, key = cv2.contourArea)
        ((x,y),r)=cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        pusat = (int(M["m10"]/M["m00"]),int(M["m01"]/M["m00"]))
        if r>10:
            cv2.circle(frame, (int(x), int(y)), int(r), (0,0,255), 2)
            cv2.circle(frame, pusat, 8, (0,0,255), 2)
    pts.appendleft(pusat)

    for i in range(1, len(pts)):
        if pts[i-1] is None or pts[i] is None:
            continue
        tebal = int(np.sqrt(args["buffer"]/float(i+1)*2.5))
        cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), tebal)
    cv2.imshow("Frame", frame)

    if cv2.waitKey(1)==27:
        break

kamera.release()
cv2.destroyAllWindows()
