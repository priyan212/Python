import cv2 as cv
import numpy as np
blank= np.zeros((500,500,3),dtype='uint8')
cv.rectangle(blank,(0,0),(250,250),(0,255,0),thickness=-1)
cv.circle(blank,(250,250),40,(0,0,255),thickness=2)
cv.line(blank,(0,0),(500,500),(255,255,255),thickness=4)
cv.putText(blank,"hello",(5,250),cv.FONT_HERSHEY_TRIPLEX,1.0,(255,255,255),thickness=2)
cv.imshow("Drawing",blank)
cv.waitKey(0)
cv.destroyAllWindows()
