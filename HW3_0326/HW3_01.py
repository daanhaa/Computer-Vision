#소벨 에지 검출 및 결과 시각화
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# 이미지 불러오기
img = cv.imread('edgeDetectionImage.jpg')  
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# Sobel 에지 검출
sobel_x = cv.Sobel(gray, cv.CV_64F, 1, 0, ksize=3)
sobel_y = cv.Sobel(gray, cv.CV_64F, 0, 1, ksize=3)

# 에지 강도 계산
magnitude = cv.magnitude(sobel_x, sobel_y)
magnitude_uint8 = cv.convertScaleAbs(magnitude)

# 시각화
plt.subplot(1, 2, 1)
plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
plt.title('Original Image')

plt.subplot(1, 2, 2)
plt.imshow(magnitude_uint8, cmap='gray')
plt.title('Edge Magnitude (Sobel)')
plt.show()
