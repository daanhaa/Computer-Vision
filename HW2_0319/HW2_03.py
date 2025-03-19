import cv2 as cv
import numpy as np

# 이미지 불러오기
image = cv.imread('tree.png')

# 이미지 크기
rows, cols = image.shape[:2]

# 45도 회전 변환 행렬 생성 (중심: (cols/2, rows/2))
rotation_matrix = cv.getRotationMatrix2D((cols/2, rows/2), 45, 1)

# 이미지 회전
rotated_image = cv.warpAffine(image, rotation_matrix, (cols, rows), flags=cv.INTER_LINEAR)

# 1.5배 확대
scaled_image = cv.resize(rotated_image, (int(cols*1.5), int(rows*1.5)), interpolation=cv.INTER_LINEAR)

# 결과 비교 출력
cv.imshow('Original', image)
cv.imshow('Rotated & Scaled', scaled_image)
cv.waitKey(0)
cv.destroyAllWindows()
