import numpy as np
import cv2

classifer = cv2.CascadeClassifier('Classifier/haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

while True:
    _,frame = cap.read()

    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    faces = classifer.detectMultiScale(gray,1.5,5)

    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),3)

    cv2.imshow("Video",frame)

    k = cv2.waitKey(1)

    if k==27:
        break
cap.release()
cv2.destroyAllWindows()
