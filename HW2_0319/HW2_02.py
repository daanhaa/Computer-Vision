#모폴로지 연산 적용
import cv2 as cv
import numpy as np

# 이미지 불러오기 및 그레이스케일 변환
image = cv.imread('JohnHancocksSignature.png', cv.IMREAD_GRAYSCALE)

# 이진화 적용
_, binary_image = cv.threshold(image, 127, 255, cv.THRESH_BINARY)

# 5x5 커널 생성
kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))

# 모폴로지 연산 적용
dilation = cv.morphologyEx(binary_image, cv.MORPH_DILATE, kernel)  # 팽창
erosion = cv.morphologyEx(binary_image, cv.MORPH_ERODE, kernel)  # 침식
opening = cv.morphologyEx(binary_image, cv.MORPH_OPEN, kernel)  # 열림 (침식 후 팽창)
closing = cv.morphologyEx(binary_image, cv.MORPH_CLOSE, kernel)  # 닫힘 (팽창 후 침식)

# 결과를 한 화면에 표시
result = np.hstack([binary_image, dilation, erosion, opening, closing])

# 창 제목 및 출력
cv.imshow('Morphology Operations (Original | Dilation | Erosion | Opening | Closing)', result)

# 키 입력 대기 후 창 닫기
cv.waitKey(0)
cv.destroyAllWindows()
