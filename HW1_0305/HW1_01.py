import cv2 as cv
import sys
import numpy as np

img = cv.imread('soccer.jpg')

if img is None :
    sys.exit('파일 없음')

#cv.cvtColor()함수로 그레이스케일로 변환
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

gray_bgr = cv.cvtColor(gray, cv.COLOR_GRAY2BGR)

#np.hstack()으로 원본이미지와 그레이스케일 이미지 가로로 연결
combined = np.hstack((img, gray_bgr))

cv.imshow('Original and Grayscale', combined)

cv.waitKey(0)
cv.destroyAllWindows()
