import cv2 as cv
import matplotlib.pyplot as plt

# 이미지 불러오기
img1 = cv.imread('mot_color70.jpg')
img2 = cv.imread('mot_color83.jpg')
gray1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
gray2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

# SIFT 특징점 추출
sift = cv.SIFT_create()
kp1, des1 = sift.detectAndCompute(gray1, None)
kp2, des2 = sift.detectAndCompute(gray2, None)

# 매칭기 생성 (BFMatcher 사용)
bf = cv.BFMatcher(cv.NORM_L2, crossCheck=True)
matches = bf.match(des1, des2)

# 거리 기준 정렬
matches = sorted(matches, key=lambda x: x.distance)

# 매칭 시각화 (상위 50개)
img_matches = cv.drawMatches(img1, kp1, img2, kp2, matches[:50], None, flags=2)

# 출력
plt.figure(figsize=(14, 7))
plt.title("SIFT 특징점 매칭 결과 (Top 50)")
plt.imshow(cv.cvtColor(img_matches, cv.COLOR_BGR2RGB))
plt.axis('off')
plt.show()
