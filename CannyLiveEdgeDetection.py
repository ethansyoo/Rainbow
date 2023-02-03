import cv2
pic = cv2.VideoCapture(0)
while(True):
    ret, frame = pic.read() #booleans
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #convert the video from BGR to gray
    frame = cv2.GaussianBlur(frame, (7, 7), 1.41) #gaussian blur with a 7 by 7 pixel and sigma x of 1.41
    edge = cv2.Canny(frame, 25, 75) #implements the Canny algorithm to find edges outputs binary image
    cv2.imshow('Canny', cv2.flip(edge, 1))
    if cv2.waitKey(20) == ord('q'):
        break