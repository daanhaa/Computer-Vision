import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# 이미지 불러오기 (그레이스케일 변환)
image = cv.imread('mistyroad.jpg', cv.IMREAD_GRAYSCALE)

# 이진화 적용
_, binary_image = cv.threshold(image, 127, 255, cv.THRESH_BINARY)

# 히스토그램 계산 (그레이스케일 & 이진화)
gray_hist = cv.calcHist([image], [0], None, [256], [0, 256])  # 그레이스케일 히스토그램
binary_hist = cv.calcHist([binary_image], [0], None, [256], [0, 256])  # 이진화 히스토그램

# 결과 출력 (이미지 | 히스토그램 나란히 배치)
plt.figure(figsize=(12, 5))

# 그레이스케일 이미지 출력
plt.subplot(1, 4, 1)
plt.imshow(image, cmap='gray')
plt.title('Grayscale Image')
plt.axis('off')

# 그레이스케일 히스토그램 출력 (옆에 배치)
plt.subplot(1, 4, 2)
plt.plot(gray_hist, color='black')
plt.title('Grayscale Histogram')

# 이진화된 이미지 출력
plt.subplot(1, 4, 3)
plt.imshow(binary_image, cmap='gray')
plt.title('Binary Image')
plt.axis('off')

#이진화된 이미지 히스토그램 출력 (옆에 배치)
plt.subplot(1, 4, 4)
plt.plot(binary_hist, color='black')
plt.title('Binary Image Histogram')

# 그래프 출력
plt.tight_layout()
plt.show()

# 이미지 출력 (OpenCV)
cv.imshow('Grayscale Image', image)
cv.imshow('Binary Image', binary_image)
cv.waitKey(0)
cv.destroyAllWindows()
