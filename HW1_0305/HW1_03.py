import cv2 as cv
import numpy as np

# 전역 변수 설정
drawing = False  # 드래그 상태 확인
ix, iy = -1, -1  # 시작 좌표
roi = None  # ROI 영역 저장

# 마우스 콜백 함수 정의
def draw_rectangle(event, x, y, flags, param):
    global ix, iy, drawing, roi

    if event == cv.EVENT_LBUTTONDOWN:  # 마우스 왼쪽 버튼 클릭 시
        drawing = True
        ix, iy = x, y

    elif event == cv.EVENT_MOUSEMOVE:  # 마우스 이동 시
        if drawing:
            img_copy = img.copy()
            cv.rectangle(img_copy, (ix, iy), (x, y), (0, 255, 0), 2)
            cv.imshow('Image', img_copy)

    elif event == cv.EVENT_LBUTTONUP:  # 마우스 왼쪽 버튼 떼기
        drawing = False
        roi = img[iy:y, ix:x]
        cv.imshow('ROI', roi)

img = cv.imread('soccer.jpg')
if img is None:
    print("이미지를 불러올 수 없습니다.")
    exit()

cv.imshow('Image', img)
cv.setMouseCallback('Image', draw_rectangle)

while True:
    key = cv.waitKey(1) & 0xFF

    if key == ord('r'):  # r 키를 누르면 초기화
        img = cv.imread('image.jpg')
        cv.imshow('Image', img)
        if roi is not None:
            cv.destroyWindow('ROI')
        roi = None

    elif key == ord('s') and roi is not None:  # s 키를 누르면 ROI 저장
        cv.imwrite('roi.png', roi)
        print("ROI가 roi.png로 저장되었습니다.")

    elif key == ord('q'):  # q 키를 누르면 종료
        break

cv.destroyAllWindows()
