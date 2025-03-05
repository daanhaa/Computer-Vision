import cv2 as cv
import numpy as np

# 웹캠 열기
cap = cv.VideoCapture(0)

# 웹캠이 정상 작동하지 않을 때 예외 처리
if not cap.isOpened():
    print("웹캠을 열 수 없습니다.")
    exit()

while True:
    # 프레임 읽기
    ret, frame = cap.read()
    if not ret:
        print("프레임을 가져올 수 없습니다.")
        break

    # BGR → Grayscale 변환
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    edges = cv.Canny(gray, 50, 150)

    #BGR 형식으로 변환 (hstack 사용을 위해)
    edges_bgr = cv.cvtColor(edges, cv.COLOR_GRAY2BGR)

    # 원본과 가로로 이어 붙이기
    combined = np.hstack((frame, edges_bgr))

    # 결과 출력
    cv.imshow('Original and Canny Edge Detection', combined)

    # q 키를 누르면 종료
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# 자원 해제
cap.release()
cv.destroyAllWindows()
