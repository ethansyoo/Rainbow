import cv2
video = cv2.VideoCapture(0)
while video.isOpened():
    run, img1 = video.read() #reads images
    run, img2 = video.read() #reads comparison image
    diff = cv2.absdiff(img1, img2) #calculates absolute difference between both images since the images are read as arrays
    diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY) #converts BGR color to gray scale
    diff_blur = cv2.GaussianBlur(diff_gray, (5, 5), 0) #removes small noise, and blurs edges given the second param (5 by 5 pixels), and the sigma x which is 0
    run, thresh = cv2.threshold(diff_blur, 20, 255, cv2.THRESH_BINARY) #if the pixel differentiation is larger than the given threshold, then max value (255), otherwise 0
    dilate = cv2.dilate(thresh, None, iterations = 3) #dilates the given array 3 iterations
    contours, hierarchy = cv2.findContours(dilate, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) #finds and connects the pixels that are changed (from 255 to 0 or 0 to 255)
    for contour in contours: #for each contour we draw a rectangle
        x, y, w, h = cv2.boundingRect(contour) 
        if cv2.contourArea(contour) > 10000: #if the area of the given contour rectangle is greater than 10000, then draw the rectangle
            cv2.rectangle(img1, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.imshow("Motion", cv2.flip(img1, 1))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        video.release()
        break