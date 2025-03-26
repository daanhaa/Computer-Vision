# HW3 - Edge & Region  
## 01. 소벨 에지 검출 및 결과 시각화

### 🔹 **수행 과정**  

1️⃣ 이미지를 **Grayscale(흑백)** 으로 변환   <br>
2️⃣ **Sobel 필터**를 사용하여 X축과 Y축 방향의 에지를 각각 검출  <br>
3️⃣ X, Y 방향 에지로부터 **에지 강도(Edge Magnitude)** 계산  <br>
4️⃣ 원본 이미지와 결과를 **matplotlib을 사용해 나란히 시각화** 
<br>

### ✅ **구현 요구사항**
💡 **OpenCV 및 matplotlib을 이용한 영상 처리 기법 구현**

| 기능 | 함수 | 설명 |
|------|------|------|
| 이미지 불러오기 | `cv.imread()` | 파일에서 이미지 읽기 |
| Grayscale 변환 | `cv.cvtColor()` | BGR → GRAY로 색 공간 변환 |
| 에지 검출 (X, Y) | `cv.Sobel()` | 각각 X축, Y축 방향 경계 검출 |
| 에지 강도 계산 | `cv.magnitude()` | √(Sobel_x² + Sobel_y²) 계산 |
| 정규화 | `cv.convertScaleAbs()` | float 결과를 8비트 정수로 변환 |
| 시각화 | `matplotlib.pyplot` | 원본 및 결과 이미지 출력 |

<br>

### 🔎 **구현 힌트**

- `cv.Sobel()` 사용 시 `ksize=3` 또는 `5`로 설정 (기본은 3 추천)  
- `cv.Sobel(gray, cv.CV_64F, dx, dy, ksize=3)`  
  → `dx=1, dy=0`: X축 방향, `dx=0, dy=1`: Y축 방향  
- `cv.magnitude()`를 사용해 X, Y 방향 경계 크기를 결합  
- `cv.convertScaleAbs()`는 결과 이미지를 0~255 범위로 스케일링  
- 시각화 시 `plt.imshow(..., cmap='gray')` 를 통해 **흑백 출력**  
- `plt.subplot()`을 사용하면 **여러 이미지를 나란히 출력** 가능

<br>

---
### **📌 코드 설명**
```python
img = cv.imread('edgeDetectionImage.jpg')  
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
```
- **이미지 불러오기 및 그레이스케일 변환** 
- cv.imread()로 컬러 이미지 불러옴
- cv.cvtColor()를 통해 Grayscale(흑백) 으로 변환
- 에지 검출은 밝기 차이(gradient) 를 기반으로 하므로, 흑백 이미지 필요

```python
sobel_x = cv.Sobel(gray, cv.CV_64F, 1, 0, ksize=3)
sobel_y = cv.Sobel(gray, cv.CV_64F, 0, 1, ksize=3)
```
-  **Sobel 필터 적용**
- cv.Sobel() 함수는 이미지의 기울기(Gradient)를 구해 경계 강조
- sobel_x: X축 방향 미분 → 수직 경계 강조
- sobel_y: Y축 방향 미분 → 수평 경계 강조
- cv.CV_64F: 더 정밀한 계산을 위해 64비트 float 타입 사용
- ksize=3: 3x3 Sobel 커널 적용

```python
magnitude = cv.magnitude(sobel_x, sobel_y)
magnitude_uint8 = cv.convertScaleAbs(magnitude)
```
- **에지 강도 계산 및 정규화**
- > cv.magnitude()를 사용해 X, Y 방향 미분값으로부터 Gradient Magnitude(에지 강도) 계산 <br>
  > 수식: √(sobel_x² + sobel_y²)
- cv.convertScaleAbs()는 float 값을 0~255 범위의 uint8 타입으로 변환

<br>

### 📌 코드 흐름 요약

1. 이미지 로딩 및 Grayscale 변환  
2. X, Y 방향으로 Sobel 연산자 적용  
3. `cv.magnitude()`로 에지 강도 계산  
4. `cv.convertScaleAbs()`로 시각화용 정규화  
5. `matplotlib.pyplot`으로 이미지 비교 출력

<br>

---

### 🖼 결과화면 
![image](https://github.com/user-attachments/assets/3685e35a-2f80-4873-89f6-c09fff5fe3f8)


### 🔗 Github 링크
[HW3_01.py](https://github.com/daanhaa/Computer-Vision/blob/main/HW3_0326/HW3_01.py)

---



## 02. 캐니 에지 및 허프 변환을 이용한 직선 검출

### 🔹 **수행 과정**

1️⃣ 이미지를 **Grayscale(흑백)** 으로 변환   <br>
2️⃣ `cv.Canny()`를 사용해 **에지 맵(Edge Map)** 생성  <br>
3️⃣ `cv.HoughLinesP()`를 통해 **허프 변환 기반 직선 검출**  <br>
4️⃣ 검출된 직선을 **원본 이미지에 빨간색 선**으로 표시  <br>
5️⃣ `matplotlib`을 사용해 **원본 이미지와 결과 이미지 시각화**<br>

<br>

### ✅ **구현 요구사항**
💡 **OpenCV 및 matplotlib을 이용한 직선 검출 구현**

| 기능 | 함수 | 설명 |
|------|------|------|
| 에지 맵 생성 | `cv.Canny()` | 이미지의 경계선(에지)을 검출 |
| 직선 검출 | `cv.HoughLinesP()` | 확률적 허프 변환을 통해 직선 검출 |
| 선 그리기 | `cv.line()` | 검출된 직선을 이미지에 시각화 |
| 시각화 | `matplotlib.pyplot` | 원본 및 결과 이미지 출력 비교 |


### **구현 힌트**

- `cv.Canny(gray, 100, 200)` → 임계값 설정: 약한 에지 vs 강한 에지  
- `cv.HoughLinesP()` 파라미터 예시:
  - `rho=1`: 거리 해상도 (픽셀 단위)  
  - `theta=np.pi/180`: 각도 해상도 (1도)  
  - `threshold=100`: 직선으로 판단할 최소 점 수  
  - `minLineLength=50`: 최소 직선 길이  
  - `maxLineGap=10`: 선분 사이 연결 허용 거리  
- `cv.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)`  
  → 빨간색 선으로 시각화 (BGR = (0, 0, 255), 두께 2)
- `plt.subplot()`으로 **원본 & 직선 검출 결과**를 나란히 비교 가능


### **📌 코드 설명**
```python
img = cv.imread('다보탑.jpg')  
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
```
- **이미지 불러오기 및 그레이스케일 변환**
- cv.imread()로 컬러 이미지를 불러옴
- cv.cvtColor()를 사용해 Grayscale(흑백)로 변환


```python
edges = cv.Canny(gray, 100, 200)

```
- **Canny 에지 검출**
- cv.Canny()는 이미지에서 경계를 감지하는 알고리즘
- 100: 낮은 임계값 (약한 에지)
- 200: 높은 임계값 (강한 에지)
- 두 임계값 사이의 연결된 에지는 유지됨 → 이진 에지 맵 생성

```python
lines = cv.HoughLinesP(edges, 1, np.pi / 180, threshold=100,minLineLength=50, maxLineGap=10)
```

- **허프 직선 검출 (Probabilistic Hough Transform)**
- edges: Canny 결과를 입력
- rho=1: 거리 해상도 (1픽셀 단위)
- theta=np.pi/180: 각도 해상도 (1도 단위)
- threshold=100: 직선으로 인식할 최소 점 개수
- minLineLength=50: 직선으로 인정할 최소 길이
- maxLineGap=10: 선분 간 최대 연결 거리


```python
line_img = img.copy()
if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv.line(line_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
```

- **검출된 직선을 이미지에 시각화**
- 원본 이미지 복사본(line_img)에 선을 그림
  > cv.line()으로 선분 시각화 <br>
  > (x1, y1) ~ (x2, y2) 좌표에 선 그리기 <br>
  > 색상: 빨간색 (BGR: (0, 0, 255)) <br>
  > 두께: 2px <br>

<br>

### 📌 코드 흐름 요약

1. 이미지 로드 및 Grayscale 변환  
2. `cv.Canny()`를 사용해 에지 맵 생성  
3. `cv.HoughLinesP()`로 직선 좌표 검출  
4. `cv.line()`으로 선을 시각화  
5. `matplotlib.pyplot`을 이용해 출력 결과 비교

---

### 🖼 결과화면
![image](https://github.com/user-attachments/assets/2db10730-ee26-470b-85c2-d6fd0ca7a054)


### 🔗 Github 링크

[HW3_02.py](https://github.com/daanhaa/Computer-Vision/blob/main/HW3_0326/HW3_02.py)

---

## 03. GrabCut을 이용한 대화식 영역 분할 및 객체 추출

---

### 🔹 **수행 과정**

1️⃣ 사용자가 지정한 **사각형 영역(Rect)** 을 기반으로 GrabCut 알고리즘 수행  <br>
2️⃣ GrabCut 결과를 **마스크 형태로 시각화**  <br>
3️⃣ 원본 이미지에서 **배경을 제거하고 객체만 추출**  <br>
4️⃣ `matplotlib`을 이용해 **원본, 마스크, 추출 결과 이미지를 나란히 시각화**<br>

<br>

### ✅ **구현 요구사항**

💡 **OpenCV의 GrabCut 알고리즘을 이용한 반자동 영역 분할 구현**

| 기능 | 함수 | 설명 |
|------|------|------|
| 이미지 불러오기 | `cv.imread()` | 파일에서 이미지 읽기 |
| 초기 마스크 생성 | `np.zeros()` | GrabCut용 마스크 초기화 |
| GrabCut 수행 | `cv.grabCut()` | 사각형 기반 객체 분리 수행 |
| 마스크 후처리 | `np.where()` | 전경만 추출하는 최종 마스크 생성 |
| 객체 추출 | `img * mask[:, :, np.newaxis]` | 전경만 남기고 배경 제거 |
| 시각화 | `matplotlib.pyplot` | 이미지 3개를 나란히 출력 |

<br>

### 🔎 **구현 힌트**

- GrabCut에서 필요한 모델 초기화:
```python
  bgdModel = np.zeros((1, 65), np.float64)
  fgdModel = np.zeros((1, 65), np.float64)
```
- np.where()를 통해 전경인 부분만 1, 나머지는 0으로 설정
- plt.subplot(1, 3, N) 으로 이미지 3개를 가로로 정렬하여 출력

<br>

### **📌 코드 설명**
```python
img = cv.imread('grabcut_sample.jpg')
```
- **이미지 불러오기**
- cv.imread()로 컬러 이미지 로딩

```python
mask = np.zeros(img.shape[:2], np.uint8)
bgdModel = np.zeros((1, 65), np.float64)
fgdModel = np.zeros((1, 65), np.float64)
rect = (50, 50, 300, 300)
```
- **GrabCut 수행을 위한 초기 설정**
- 빈 마스크(mask)는 이미지와 동일한 크기의 2D 배열로 생성
- bgdModel, fgdModel은 GrabCut 내부 계산에 사용하는 임시 모델
- rect는 사용자가 지정하는 전경 영역

```python
cv.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv.GC_INIT_WITH_RECT)
```
- **GrabCut 알고리즘 수행**
- 지정된 rect를 기준으로 객체(전경)와 배경을 분리
- 5번 반복하며 최적화 수행
- cv.GC_INIT_WITH_RECT: 초기 사각형 기반 분할 수행

<br>

### 📌 코드 흐름 요약
- 이미지 로드 및 마스크/모델 초기화
- GrabCut 알고리즘 수행 (초기 rect 기반)
- np.where()로 전경 마스크 생성
- 객체만 남기고 배경 제거
- matplotlib.pyplot으로 원본/마스크/결과 이미지 비교 출력

---

### 🖼 결과화면 
![image](https://github.com/user-attachments/assets/8fe238ee-2c1e-480b-8d62-e6a373a8d32d)



### 🔗 Github 링크
[HW3_03.py](https://github.com/daanhaa/Computer-Vision/blob/main/HW3_0326/HW3_03.py)

---
