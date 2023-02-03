import cv2
cap = cv2.VideoCapture(0) #opens webcam on device
cap.set(3, 640) #sets frame width to 640
cap.set(4, 480) #sets frame height to 480
while cap.isOpened(): #while cam is on
    run, img = cap.read() #cap.read() returns bool value in terms of the video frame
    if run: #while there is a frame
        cv2.imshow("Live", cv2.flip(img, 1)) #show the frame
        if cv2.waitKey(1) & 0xFF == ord('q'): #once 'p' is pressed, break
            break
