import cv2 as cv
import numpy as np

# 이미지 불러오기 및 그레이스케일 변환
image = cv.imread('tree.png', cv.IMREAD_GRAYSCALE)

# 오츠 알고리즘을 이용한 이진화
_, binary_image = cv.threshold(image, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

# 5x5 커널 생성
kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))

# 모폴로지 연산 적용
dilation = cv.morphologyEx(binary_image, cv.MORPH_DILATE, kernel)  # 팽창
erosion = cv.morphologyEx(binary_image, cv.MORPH_ERODE, kernel)  # 침식
opening = cv.morphologyEx(binary_image, cv.MORPH_OPEN, kernel)  # 열림 (침식 후 팽창)
closing = cv.morphologyEx(binary_image, cv.MORPH_CLOSE, kernel)  # 닫힘 (팽창 후 침식)

# 결과를 하나의 화면에 붙이기 (가로로 정렬)
result = np.hstack([binary_image, dilation, erosion, opening, closing])

# 현재 화면 크기 가져오기 (디스플레이 해상도에 맞게 자동 조절)
screen_width, screen_height = 1280, 720  # 기본 해상도 설정
window_name = "Morphology Operations"

# 창을 생성하여 화면 크기를 확인 (OpenCV 4.5 이상에서 사용 가능)
cv.namedWindow(window_name, cv.WINDOW_NORMAL)

# 이미지 크기 조정 (화면 크기에 맞추되 비율 유지)
scale_factor = min(screen_width / result.shape[1], screen_height / result.shape[0])
new_width = int(result.shape[1] * scale_factor)
new_height = int(result.shape[0] * scale_factor)

resized_result = cv.resize(result, (new_width, new_height), interpolation=cv.INTER_AREA)

# 창 제목 및 출력
cv.imshow(window_name, resized_result)

# 키 입력 대기 후 창 닫기
cv.waitKey(0)
cv.destroyAllWindows()
