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

<br>
<br>

### 구현 결과
![image](https://github.com/user-attachments/assets/4a0c53e0-2fc7-4af1-9500-17d0e59eff34)

#### GITHUB
#### [HW4_01파일로 이동](https://github.com/daanhaa/Computer-Vision/blob/main/HW4_0402/HW4_01.py)
---

## 02-1. SIFT 특징점 매칭 - BFMatcher (`HW4_02.py`)

### 🔹 수행 과정

1️⃣ 두 이미지를 **Grayscale**로 변환  
2️⃣ 두 이미지에서 **SIFT 특징점** 추출  
3️⃣ `BFMatcher`를 사용하여 특징점 매칭 수행  
4️⃣ 거리 기준으로 정렬 후, 매칭 결과를 시각화

---

### 📌 코드 설명

```python
bf = cv.BFMatcher(cv.NORM_L2, crossCheck=True)
matches = bf.match(des1, des2)
matches = sorted(matches, key=lambda x: x.distance)
```

####  BFMatcher를 이용한 특징점 매칭

- `cv.BFMatcher()`는 Brute-Force 방식으로 모든 기술자(descriptor) 쌍의 거리를 계산합니다.
- `cv.NORM_L2`는 SIFT와 같은 float 기반 기술자에 적합한 거리 계산 방식입니다.
- `crossCheck=True` 설정 시, 양방향 매칭이 일치하는 경우만 유효한 매칭으로 인정 → 잘못된 매칭 제거 효과가 있음.
- `sorted(..., key=lambda x: x.distance)`는 거리(distance)가 가장 짧은 것부터 나열하여 유사도가 높은 순으로 정렬합니다.

```python
img_matches = cv.drawMatches(img1, kp1, img2, kp2, matches[:50], None, flags=2)
```

#### 🖼️ 매칭 결과 시각화

- `cv.drawMatches()`는 두 이미지 사이의 매칭된 특징점들을 선으로 연결하여 시각화합니다.
- `matches[:50]`은 정렬된 결과 중 상위 50개만 시각화하여 복잡하지 않고 명확한 비교가 가능하게 합니다.
- `flags=2`는 키포인트의 크기, 방향 등을 생략한 간단한 형태의 매칭 시각화 옵션입니다.

<br>
<br>

### BFMatcher 구현 결과
![image](https://github.com/user-attachments/assets/c46979ee-4c91-42ad-aff4-0a146683b108)

#### GITHUB
#### [HW4_02-1파일로 이동](https://github.com/daanhaa/Computer-Vision/blob/main/HW4_0402/HW4_02.py)
---


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

#### 🔍 1. SIFT 특징점 검출
```python
sift = cv.SIFT_create()
kp1, des1 = sift.detectAndCompute(gray1, None)
kp2, des2 = sift.detectAndCompute(gray2, None)
```
- `detectAndCompute()`는 각 이미지의 특징점(keypoint)과 기술자(descriptor)를 추출합니다.
- 특징점: 이미지 내에서 의미 있는 위치 (예: 코너, 윤곽 등)
- 기술자: 해당 특징점을 수치적으로 표현한 벡터


#### 🔍 2. FLANN 기반 매칭기 생성 및 설정
```python
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv.FlannBasedMatcher(index_params, search_params)
```
- FLANN(빠른 근사 최근접 이웃)은 대규모 데이터셋에 적합한 고속 매칭 알고리즘입니다.
- KD-Tree 알고리즘 기반으로 특징 벡터 간 근사 거리 계산
- `trees`: 인덱스를 구성할 트리 개수 (5~10 추천)
- `checks`: 검색 시 비교할 노드 수 (속도/정확도 트레이드오프)


#### 🔍 3. KNN 매칭 수행 및 Lowe’s Ratio Test 적용
```python
matches = flann.knnMatch(des1, des2, k=2)
```
- 각 기술자에 대해 가장 가까운 2개의 매칭점을 반환 → Ratio Test에 사용

```python
good_matches = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good_matches.append(m)
```
- **Lowe의 ratio test**: 두 최근접 매칭 간의 거리 차이를 비교해 신뢰도 높은 매칭만 필터링
- 0.75는 보통 경험적으로 가장 많이 사용하는 기준값


#### 🖼️ 4. 매칭 시각화
```python
img_matches = cv.drawMatches(img1, kp1, img2, kp2, good_matches[:50], None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
```
- 상위 50개의 좋은 매칭 결과만 시각화하여 복잡함을 줄이고 가독성을 높임
- `NOT_DRAW_SINGLE_POINTS` 옵션은 매칭된 점만 시각화하며, 단독 키포인트는 생략합니다.

```python
plt.imshow(cv.cvtColor(img_matches, cv.COLOR_BGR2RGB))
```
- OpenCV는 BGR 포맷을 사용하므로 matplotlib에서 올바르게 출력하려면 RGB로 변환 필요

---


<br>
<br>

### FLANN 구현 결과
![image](https://github.com/user-attachments/assets/818efb74-3f13-485f-8b16-f632a7e9305f)

#### GITHUB
#### [HW4_02-2파일로 이동](https://github.com/daanhaa/Computer-Vision/blob/main/HW4_0402/HW4_02FLANN.py)


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

#### 🔍 1. 호모그래피 계산 및 이미지 정합 (Warping)

```python
H, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
warped = cv.warpPerspective(img1, H, (img2.shape[1], img2.shape[0]))
```
- `cv.findHomography()`는 **두 이미지 간 변환 행렬(H)** 계산
  - `cv.RANSAC` 사용으로 이상치(outlier) 제거
- `cv.warpPerspective()`는 **이미지1을 이미지2의 관점으로 투시 변환** (정합)


#### 🔍 2. 자동 이미지 스티칭 (cv2.Stitcher)

```python
stitcher = cv.Stitcher_create()
status, stitched = stitcher.stitch([img1, img2])
```
- OpenCV의 고수준 API로 **자동 파노라마 생성 기능 제공**
- 내부적으로:
  - 특징점 검출 → 매칭 → 호모그래피 계산 → 블렌딩까지 모두 자동 수행
- `status == cv.Stitcher_OK`일 경우 정상적으로 스티칭 완료

#### 🔍 3. Overlap 시각화
```python
overlap = cv.addWeighted(warped_rgb, 0.5, img2_rgb, 0.5, 0)
```
- **cv.addWeighted()**를 통해 정합된 이미지와 기준 이미지의 겹치는 영역을 반투명하게 출력
- Overlap 시각화는 두 이미지의 정합 정확도 시각적 검증에 매우 유용함
- 일반적으로 Homography의 성능을 직관적으로 확인할 수 있는 좋은 방법


#### 🖼️ 4. 결과 시각화

```python
plt.subplot(1, 3, N)
plt.imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB))
```
- **세 개의 이미지 시각화**:
  > 기준 이미지 (img2) <br>
  > 정합된 이미지 (Homography)  <br>
  > Stitcher 스티칭 이미지 <br>
- BGR → RGB 색공간 변환 후 matplotlib으로 출력
- `plt.tight_layout()`으로 간격 자동 조정하여 깔끔한 레이아웃 정렬

---

<br>
<br>

### 구현 결과
![image](https://github.com/user-attachments/assets/1570f108-3f44-493b-940c-39fd9a789ad4)

![image](https://github.com/user-attachments/assets/5812f6b8-38a6-46d4-9984-584b67c5eb6a)

![image](https://github.com/user-attachments/assets/8ce4e3a0-314b-4dec-a4e6-3003dceb4aa4)



#### GITHUB
#### [HW4_03파일로 이동](https://github.com/daanhaa/Computer-Vision/blob/main/HW4_0402/HW4_03.py)


---

#### +++ 특징점 원 크기가 달라지는 이유
- SIFT 특징점 위에 크기가 다른 원들이 그려진 이유는 SIFT 특징점 시각화 시 cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS 옵션을 사용했기 때문
1. SIFT는 다양한 스케일 공간에서 특징점을 검출함 (Gaussian Pyramid)
2. 각 특징점은 자신이 검출된 스케일 정보를 포함
3. DRAW_RICH_KEYPOINTS는 이 스케일을 원 크기로 반영해서 그림
4. 따라서 큰 원은 큰 구조나 멀리 있는 물체, 작은 원은 작고 세밀한 부분을 의미함


