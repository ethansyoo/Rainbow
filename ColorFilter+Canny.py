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
    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY) #converts HSV to gray
    blur = cv2.GaussianBlur(gray, (5, 5), 1.41) #blurs edges to make for better edge detection
    edge = cv2.Canny(result, 50, 150) #detects edges of only those within the color parameter outputs binary image
    indices = np.where(edge != [0]) #detects edges that are left on the resulting image
    coordinates = zip(indices[0], indices[1]) #zips all the indices together
    print(list(coordinates)[1]) #calls the list version of each coordinate that is considered as an edge
    image = cv2.flip(edge, 1) #displays the resulting color filter
    image = cv2.line(image,list(coordinates)[0],list(coordinates)[1],(255,0,0),5)
    cv2.imshow("Result", image)
    if cv2.waitKey(1) & 0xFF == ord('q'): #if 'q' key is pressed, close the prompt
            cv2.destroyAllWindows()
            video.release()
            break
