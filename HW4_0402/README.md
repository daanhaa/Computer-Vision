# HW4 - Local Feature (지역특징)

---

## 01. SIFT 특징점 검출 및 시각화 (`HW4_01.py`)

### 🔹 수행 과정

1️⃣ 이미지를 **Grayscale(흑백)**으로 변환  
2️⃣ **SIFT 알고리즘**을 사용하여 특징점을 추출  
3️⃣ 검출된 특징점을 이미지 위에 시각화  
4️⃣ 원본 이미지와 특징점 결과를 matplotlib으로 나란히 시각화

### ✅ 구현 요구사항

| 기능 | 함수 | 설명 |
|------|------|------|
| 이미지 불러오기 | `cv.imread()` | 이미지 파일 읽기 |
| Grayscale 변환 | `cv.cvtColor()` | BGR → GRAY로 변환 |
| SIFT 특징점 검출 | `cv.SIFT_create().detectAndCompute()` | 특징점 및 기술자 계산 |
| 특징점 시각화 | `cv.drawKeypoints()` | 특징점 이미지에 시각화 |
| 시각화 | `matplotlib.pyplot` | 이미지 출력 |

---

### **📌 코드 설명**
```python
img = cv.imread('mot_color70.jpg')  
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
```
- **이미지 불러오기 및 그레이스케일 변환**
- cv.imread()로 컬러 이미지를 불러옴
- cv.cvtColor()를 통해 Grayscale(흑백)으로 변환
- SIFT 특징점 검출을 위해 흑백 이미지 필요

```python
sift = cv.SIFT_create(nfeatures=300)
keypoints, descriptors = sift.detectAndCompute(gray, None)
```
- **SIFT 특징점 추출**
- 특징점 수 제한을 위해 nfeatures 파라미터 사용

```python
img_keypoints = cv.drawKeypoints(img, keypoints, None, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
```
- **특징점 시각화**
- 특징점의 크기와 방향을 이미지 위에 표현

---

## 02-1. SIFT 특징점 매칭 - BFMatcher (`HW4_02.py`)

### 🔹 수행 과정

1️⃣ 두 이미지를 **Grayscale**로 변환  
2️⃣ 두 이미지에서 **SIFT 특징점** 추출  
3️⃣ `BFMatcher`를 사용하여 특징점 매칭 수행  
4️⃣ 거리 기준으로 정렬 후, 매칭 결과를 시각화

---

### **📌 코드 설명**
```python
bf = cv.BFMatcher(cv.NORM_L2, crossCheck=True)
matches = bf.match(des1, des2)
matches = sorted(matches, key=lambda x: x.distance)
```
- **BFMatcher를 이용한 특징점 매칭**
- 거리(distance)를 기준으로 매칭 결과 정렬

```python
img_matches = cv.drawMatches(img1, kp1, img2, kp2, matches[:50], None, flags=2)
```
- **매칭 결과 시각화**
- 좋은 매칭을 상위 50개로 제한하여 시각화

---

## 02-2. SIFT 특징점 매칭 - FLANN (`HW4_02_FLANN.py`)

### 🔹 수행 과정

1️⃣ 두 이미지를 **Grayscale**로 변환  
2️⃣ **SIFT 특징점** 추출  
3️⃣ FLANN 기반 매칭기 사용하여 특징점 매칭  
4️⃣ Lowe의 Ratio Test를 통해 좋은 매칭점 선별  
5️⃣ 매칭 결과를 시각화

---

### **📌 코드 설명**
```python
flann = cv.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des1, des2, k=2)
```
- **FLANN 매칭**
- knnMatch()로 두 개의 가장 가까운 이웃을 찾음

```python
good_matches = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good_matches.append(m)
```
- **좋은 매칭점 선별 (Lowe's ratio test)**
- 정확도가 높은 매칭점만 선택

---

## 03. 호모그래피를 이용한 이미지 정합 (`HW4_03.py`)

### 🔹 수행 과정

1️⃣ 두 이미지를 **Grayscale**로 변환  
2️⃣ **SIFT 특징점** 추출 및 **기술자** 계산  
3️⃣ BFMatcher를 이용한 knn 매칭 및 Lowe의 ratio test 수행  
4️⃣ 매칭된 특징점을 사용하여 호모그래피 계산(RANSAC 사용)  
5️⃣ 호모그래피 행렬을 사용해 이미지 정합 및 시각화

---

### **📌 코드 설명**
```python
H, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
aligned = cv.warpPerspective(img1, H, (img2.shape[1], img2.shape[0]))
```
- **호모그래피 계산 및 이미지 정합**
- RANSAC 알고리즘을 통해 이상점의 영향을 최소화
- 호모그래피 행렬을 이용하여 이미지 정합(warp)

---