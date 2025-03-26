#캐니 에지 및 허프 변환을 이용한 직선 검출
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# 이미지 불러오기
img = cv.imread('다보탑.jpg')  
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# Canny 에지맵 생성
edges = cv.Canny(gray, 100, 200)

# 허프 직선 변환
lines = cv.HoughLinesP(edges, 1, np.pi / 180, threshold=100,
                       minLineLength=50, maxLineGap=10)

# 복사 이미지에 직선 그리기
line_img = img.copy()
if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv.line(line_img, (x1, y1), (x2, y2), (0, 0, 255), 2)

# 시각화
plt.subplot(1, 2, 1)
plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
plt.title('Original Image')

plt.subplot(1, 2, 2)
plt.imshow(cv.cvtColor(line_img, cv.COLOR_BGR2RGB))
plt.title('Hough Lines')
plt.show()
