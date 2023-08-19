import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
cap = cv.VideoCapture(0)
img_counter = 0
a=0
b=0
Gw=0
Gh=0
face_classifier = cv.CascadeClassifier(
    cv.data.haarcascades + "haarcascade_frontalface_default.xml"
)
if not cap.isOpened():
 print("Cannot open camera")
 exit()
while True:
 # Capture frame-by-frame
 ret, frame = cap.read()
 # if frame is read correctly ret is True
 if not ret:
    print("Can't receive frame (stream end?). Exiting ...")
 face = face_classifier.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))

 for (x, y, w, h) in face:
    cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 4)
    frame=frame[x:x+w,y:y+h]
    a=x
    Gh=h
    Gw=w
    b=y


 # if len(face)==1:
 #    img_name = f'opencv_frame{img_counter}.png'
 #    img_counter=img_counter+1
 #        # saves the image as a png file
 #    cv.imwrite(img_name,frame)
 #    print('screenshot taken')
 #    break

 #cv.rectangle(face)


 cv.imshow('frame', frame)
 if cv.waitKey(1) == ord('q'):
    break
# When everything done, release the capture
print (frame.shape)
print (len(face))
print ("Value of a is ")
print (a)
print ("Value of b is ")
print (b)

cap.release()
cv.destroyAllWindows()

