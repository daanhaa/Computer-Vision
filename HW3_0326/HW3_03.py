import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# 이미지 파일 경로 설정
image_path = 'coffee cup.JPG'

# 이미지 불러오기
img = cv.imread(image_path)

# 이미지가 정상적으로 로드되었는지 확인
if img is None:
    print(f"이미지를 불러올 수 없습니다. 경로를 확인하세요: {image_path}")
else:
    # GrabCut 초기 설정
    mask = np.zeros(img.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    rect = (0, 0, 1250, 1250)  # (x, y, w, h)

    # GrabCut 수행
    cv.grabCut(img, mask, rect, bgdModel, fgdModel, 20, cv.GC_INIT_WITH_RECT)

    # 마스크 처리 (0: 배경, 1: 전경)
    mask2 = np.where((mask == cv.GC_FGD) | (mask == cv.GC_PR_FGD), 1, 0).astype('uint8')
    result = img * mask2[:, :, np.newaxis]

    # 시각화
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    plt.title('Original')

    plt.subplot(1, 3, 2)
    plt.imshow(mask, cmap='gray')
    plt.title('GrabCut Mask')

    plt.subplot(1, 3, 3)
    plt.imshow(cv.cvtColor(result, cv.COLOR_BGR2RGB))
    plt.title('Foreground Extracted')
    plt.show()
