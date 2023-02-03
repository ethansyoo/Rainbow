import cv2
import os
cascPath = os.path.dirname(cv2.__file__) + "/data/haarcascade_frontalface_alt2.xml" #pulls up the cascade file path
faceCascade = cv2.CascadeClassifier(cascPath) #calls the classifier within the file
vid = cv2.VideoCapture(0)
while True:
    _, frame = vid.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #converts image from BGR to gray
    faces = faceCascade.detectMultiScale(gray, scaleFactor = 1.1, 
                                            minNeighbors = 5, minSize = (60, 60), 
                                            flags = cv2.CASCADE_SCALE_IMAGE) 
                                            #detects objects of different sizes in input image, and is returned as a list of rectangles
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2) #draws rectangle around given pixel points
    cv2.imshow("Video", cv2.flip(frame, 1))
    if cv2.waitKey(20) == ord('q'):
        break
vid.release()
cv2.destroyAllWindows()