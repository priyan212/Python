import cv2 as cv
import numpy as np
img=cv.imread('photos/pagla painting.png')
cv.imshow("image",img)
cv.waitKey(0)
blur=cv.GaussianBlur(img,(7,7),cv.BORDER_DEFAULT)
edge=cv.Canny(blur,125,125)
cv.imshow("edge",edge)
cv.waitKey(0)
dilate=cv.dilate(img,(7,7),iterations=9)
cv.imshow("Dilated",dilate)
cv.waitKey(0)
