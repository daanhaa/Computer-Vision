import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# 이미지 불러오기
img1 = cv.imread('img1.jpg')
img2 = cv.imread('img2.jpg')
gray1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
gray2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

# SIFT 특징점 검출
sift = cv.SIFT_create()
kp1, des1 = sift.detectAndCompute(gray1, None)
kp2, des2 = sift.detectAndCompute(gray2, None)

# BFMatcher + knn
bf = cv.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)

# Lowe's ratio test 적용
good_matches = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good_matches.append(m)

# 이미지 정합 수행
warped = None
overlap = None
if len(good_matches) > 4:
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # 호모그래피 계산
    H, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)

    # 이미지 정합 (Warping)
    h, w = img1.shape[:2]
    warped = cv.warpPerspective(img1, H, (img2.shape[1], img2.shape[0]))

    # 겹치는 영역 시각화
    warped_rgb = cv.cvtColor(warped, cv.COLOR_BGR2RGB)
    img2_rgb = cv.cvtColor(img2, cv.COLOR_BGR2RGB)
    overlap = cv.addWeighted(warped_rgb, 0.5, img2_rgb, 0.5, 0)
else:
    print("특징점 매칭이 충분하지 않습니다.")

# 이미지 스티칭 (Stitcher)
stitcher = cv.Stitcher_create()
status, stitched = stitcher.stitch([img1, img2])

# 정합 결과 및 스티칭 결과 출력
plt.figure(figsize=(18, 6))

plt.subplot(1, 3, 1)
plt.title("Base Image")
plt.imshow(cv.cvtColor(img2, cv.COLOR_BGR2RGB))
plt.axis('off')

if warped is not None:
    plt.subplot(1, 3, 2)
    plt.title("Warped Image (Homography)")
    plt.imshow(warped_rgb)
    plt.axis('off')

if status == cv.Stitcher_OK:
    plt.subplot(1, 3, 3)
    plt.title("Stitched Image (Stitcher)")
    plt.imshow(cv.cvtColor(stitched, cv.COLOR_BGR2RGB))
    plt.axis('off')
else:
    print("Stitching 실패. 코드:", status)

plt.tight_layout()
plt.show()

# -----------------------
# Overlap 이미지 따로 출력
# -----------------------
if overlap is not None:
    plt.figure(figsize=(8, 6))
    plt.title("Overlap Visualization")
    plt.imshow(overlap)
    plt.axis('off')
    plt.tight_layout()
    plt.show()
