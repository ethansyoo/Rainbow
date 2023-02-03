import cv2
import os

while True:
    img = cv2.imread(os.path.expanduser("~/Downloads/blue.jpeg"))
    image = img.read()
    cv2.namedWindow("Images")
    cv2.imshow("Images", image)




# import cv2
# import numpy as np
# pic = cv2.VideoCapture(1)
# while(True):
#     run, img = pic.read() #bool value
#     hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) #convert BGR image to HSV image
#     lower = np.array([90, 50, 70]) #lower bound HSV array for blue
#     upper = np.array([128, 255, 255]) #upper bound HSV array for blue
#     mask = cv2.inRange(hsv, lower, upper) #masks everything except the hsv values within the boundaries
#     result = cv2.bitwise_and(img, img, mask = mask) #merges both pictures with the masking value
#     gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY) #converts HSV to gray
#     blur = cv2.GaussianBlur(gray, (5, 5), 1.41) #blurs edges to make for better edge detection
#     edge = cv2.Canny(result, 50, 150) #detects edges of only those within the color parameter outputs binary image
#     row_indexes, col_indexes = np.nonzero(edge)
#     cv2.imshow("Result", edge) #displays the resulting color filter
#     if cv2.waitKey(1) & 0xFF == ord('q'): #if 'q' key is pressed, close the prompt
#             cv2.destroyAllWindows()
#             pic.release()
#             break
