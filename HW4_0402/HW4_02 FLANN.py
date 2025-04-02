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

# FLANN 매칭기 파라미터 설정
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)

# FLANN 매칭기 생성
flann = cv.FlannBasedMatcher(index_params, search_params)

# knnMatch 수행 (최근접 이웃 두 개 반환)
matches = flann.knnMatch(des1, des2, k=2)

# Lowe의 ratio test로 좋은 매칭점 선택
good_matches = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good_matches.append(m)

# 매칭 결과 시각화 (좋은 매칭점 상위 50개)
img_matches = cv.drawMatches(img1, kp1, img2, kp2, good_matches[:50], None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# 결과 출력
plt.figure(figsize=(14, 7))
plt.title("FLANN 기반 SIFT 특징점 매칭 결과 (Top 50)")
plt.imshow(cv.cvtColor(img_matches, cv.COLOR_BGR2RGB))
plt.axis('off')
plt.show()
