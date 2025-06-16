import cv2 as cv
import numpy as np
img=cv.imread('photos/partial_2.png')
#cv.imshow("normal",img)
resized=cv.resize(img,(1000,500),interpolation=cv.INTER_CUBIC)
cv.waitKey(0)
crop= resized[50:400, 50:800]
cv.imshow("cropped",crop)
cv.waitKey(0)
