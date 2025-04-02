import cv2 as cv
import matplotlib.pyplot as plt

# 이미지 불러오기
img = cv.imread('mot_color70.jpg')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# SIFT 객체 생성 (nfeatures로 특징점 수 제한 가능)
sift = cv.SIFT_create(nfeatures=300)

# 특징점 검출
keypoints, descriptors = sift.detectAndCompute(gray, None)

# 특징점 시각화 (크기 및 방향 포함)
img_keypoints = cv.drawKeypoints(
    img, keypoints, None,
    flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
)

# 시각화
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Origin")
plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("SIFT feature point")
plt.imshow(cv.cvtColor(img_keypoints, cv.COLOR_BGR2RGB))
plt.axis('off')
plt.tight_layout()
plt.show()
