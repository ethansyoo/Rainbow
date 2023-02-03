import cv2
import numpy as np

video = cv2.VideoCapture(0) #turns on device camera
while(True):
    run, img = video.read() #bool value
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) #convert BGR image to HSV image
    lower = np.array([90, 50, 70]) #lower bound HSV array for blue
    upper = np.array([128, 255, 255]) #upper bound HSV array for blue
    mask = cv2.inRange(hsv, lower, upper) #masks everything except the hsv values within the boundaries
    result = cv2.bitwise_and(img, img, mask = mask) #merges both pictures with the masking value
    cv2.imshow("Result", cv2.flip(result, 1)) #displays the resulting color filter
    if cv2.waitKey(1) & 0xFF == ord('q'): #if 'q' key is pressed, close the prompt
            cv2.destroyAllWindows()
            video.release()
            break
