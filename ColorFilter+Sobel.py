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
    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    grad_x = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize = 3)
    grad_y = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize = 3)
    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)
    grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    cv2.imshow("Result", cv2.flip(grad, 1)) #displays the resulting color filter
    if cv2.waitKey(1) & 0xFF == ord('q'): #if 'q' key is pressed, close the prompt
            cv2.destroyAllWindows()
            video.release()
            break