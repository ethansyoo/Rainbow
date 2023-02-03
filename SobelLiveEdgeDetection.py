import cv2
video = cv2.VideoCapture(0)
while(True):
    ret, frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 1.41)
    grad_x = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize = 3) #must split sobel gradients into x and y
    grad_y = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize = 3)
    abs_grad_x = cv2.convertScaleAbs(grad_x) #finding the direction of most change in terms of pixels
    abs_grad_y = cv2.convertScaleAbs(grad_y)
    grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0) #adds weights to x and y, equally weighing them
    cv2.imshow('Sobel', cv2.flipi(grad, 1))
    if cv2.waitKey(20) == ord('q'):
        break

