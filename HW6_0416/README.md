# HW6 - Dynamic Vision

---


## 01. SORT 알고리즘을 활용한 다중 객체 추적기 구현

### 📘 설명

이 실습에서는 SORT 알고리즘을 사용하여 비디오에서 다중 객체를 실시간으로 추적하는 프로그램을 구현합니다.
이를 통해 객체 추적의 기본 개념과 SORT 알고리즘의 적용 방법을 학습할 수 있습니다.

### ✅ 구현 요구사항

1. **객체 검출기 구현**
- YOLOv4와 같은 사전 학습된 객체 검출 모델을 사용하여 각 프레임에서 객체를 검출합니다.
2. **SORT 추적기 초기화**
- 검출된 객체의 경계 상자를 입력으로 받아 SORT 추적기를 초기화합니다.
3. **객체 추적 유지**
- 각 프레임마다 검출된 객체와 기존 추적 객체를 연관시켜 추적을 유지합니다.
4. **결과 시각화**
- 추적된 각 객체에 고유 ID를 부여하고, 해당 ID 및 경계 상자를 비디오 프레임에 실시간으로 표시합니다.

### 💡 힌트
- **객체 검출**:
- > OpenCV의 cv2.dnn 모듈을 활용하여 YOLOv4 모델을 로드하고, 각 프레임에서 객체를 탐지할 수 있습니다.
- **SORT 알고리즘**:
- > SORT는 **칼만 필터(Kalman Filter)**와 **헝가리안 알고리즘(Hungarian Algorithm)**을 사용하여
  > 객체의 상태를 예측하고 탐지 결과와의 매칭을 수행합니다.
- **추적 성능 향상:**
- > 추적 대상의 외형 정보(appearance)를 고려하는 Deep SORT 등 확장 알고리즘도 존재하며,
  > 이들을 활용하면 보다 정교한 추적이 가능합니다.
---

### **📌 코드 설명**

### 1. 📁 라이브러리 및 모델 초기화

```python
import cv2 as cv
import numpy as np
from sort import Sort
```

- `cv2`: OpenCV 영상 처리 라이브러리  
- `numpy`: 수치 계산 라이브러리  
- `Sort`: 외부 구현된 객체 추적기

---

### 2. 📚 클래스 이름 불러오기

```python
with open("coco.names", "r") as f:
    class_names = [line.strip() for line in f.readlines()]
```

- YOLOv4가 탐지할 수 있는 클래스명들을 불러옴 (예: car, person 등)

---

### 3. ⚙️ YOLOv4 모델 로드

```python
net = cv.dnn.readNet("yolov4.weights", "yolov4.cfg")
layer_names = net.getLayerNames()
out_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
```

- YOLOv4의 구조와 사전 학습된 가중치를 로드  
- 출력 레이어 이름을 추출하여 `forward()` 결과에서 사용

---

### 4. 🛰️ SORT 추적기 초기화

```python
tracker = Sort()
```

- SORT(Simple Online and Realtime Tracking) 객체 생성  
- 객체마다 고유 ID를 부여하고 추적 상태 유지

---

### 5. 🎥 비디오 파일 열기

```python
cap = cv.VideoCapture("slow_traffic_small.mp4")
if not cap.isOpened():
    raise IOError("비디오 파일을 열 수 없습니다.")
```

- 테스트용 비디오를 불러옴  
- 파일이 열리지 않으면 예외 발생

---

### 6. 🎨 추적 ID용 색상 배열 생성

```python
colors = np.random.randint(0, 255, size=(100, 3))
```

- 각 ID에 대응되는 색상 미리 지정  
- 색상은 0~255 범위의 RGB 값으로 생성됨

---

### 7. 🔁 프레임 반복 처리

```python
while True:
    ret, frame = cap.read()
    if not ret:
        break
```

- 비디오에서 한 프레임씩 읽음  
- 더 이상 프레임이 없으면 반복 종료

---

### 8. 🧠 YOLOv4 객체 탐지 수행

```python
blob = cv.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
net.setInput(blob)
outputs = net.forward(out_layers)
```

- 프레임을 YOLO 입력 형태(blob)로 전처리  
- 정규화 후 RGB로 변환 (OpenCV는 BGR이 기본이므로 `swapRB=True`)  
- `net.forward()`로 객체 탐지 수행

---

### 9. 🚗 자동차 클래스만 필터링

```python
if confidence > 0.5 and class_names[class_id] == "car":
```

- 탐지된 객체 중 신뢰도가 50% 이상이며 `"car"`인 객체만 추적 대상  
- 중심 좌표 → 좌상단/우하단 좌표로 변환

---

### 10. 🛰️ SORT에 감지된 박스 전달

```python
dets = np.array(boxes)
tracked = tracker.update(dets)
```

- YOLO가 반환한 박스를 배열로 정리 후 SORT에 전달  
- `tracked` 결과: `[x1, y1, x2, y2, ID]` 형식으로 반환됨

---

### 11. 🖼️ 추적 결과 시각화

```python
for d in tracked:
    x1, y1, x2, y2, obj_id = d.astype(int)
    color = colors[int(obj_id) % 100]
    cv.rectangle(frame, (x1, y1), (x2, y2), color.tolist(), 2)
    cv.putText(frame, f'ID: {int(obj_id)}', (x1, y1 - 10), cv.FONT_HERSHEY_SIMPLEX, 0.6, color.tolist(), 2)
```

- 추적된 객체에 대해 사각형 박스를 그림  
- ID마다 고유한 색상을 적용  
- 사각형 상단에 `ID: n` 형식으로 ID 출력

---

### 12. 📺 화면 출력 및 종료 조건

```python
cv.imshow("YOLOv4 + SORT Tracking (Video)", frame)
if cv.waitKey(1) & 0xFF == 27:  # ESC 키
    break
```

- 결과 프레임을 실시간으로 출력  
- 사용자가 ESC 키를 누르면 종료
---

## ✅ 요약

| 항목 | 설명 |
|------|------|
| 객체 탐지 | YOLOv4 사용 (`cv.dnn`) |
| 객체 추적 | SORT 알고리즘 적용 |
| 대상 | `"car"` 클래스만 필터링 |
| 출력 | 고유 ID와 함께 경계 박스 표시 |
| 종료 | ESC 키 입력 시 종료 |


#### 구현 결과

![image](https://github.com/user-attachments/assets/a8081099-26e1-4645-8b9d-76d4cb0ffeba)

[HW6_01.py코드](https://github.com/daanhaa/Computer-Vision/blob/main/HW6_0416/HW6_01.py)

---
## 🎯 Dynamic Vision - 02  
### Mediapipe를 활용한 얼굴 랜드마크 추출 및 시각화

---

## 📘 설명

이 실습에서는 Mediapipe의 `FaceMesh` 모듈을 활용하여 **실시간 영상에서 얼굴의 468개 랜드마크를 추출**하고,  
각 랜드마크를 OpenCV를 이용해 점으로 시각화하는 프로그램을 구현합니다.

---

## ✅ 구현 요구사항

1. `FaceMesh` 모듈을 사용하여 얼굴 랜드마크 검출기를 초기화
2. OpenCV를 이용해 웹캠에서 실시간 영상 캡처
3. 검출된 얼굴 랜드마크를 `cv2.circle()`을 사용해 점으로 표시
4. ESC 키를 누르면 프로그램 종료

---

## 💡 힌트

- `mediapipe.solutions.face_mesh`를 사용하여 얼굴 랜드마크 검출기 생성
- `landmark.x`, `landmark.y`는 정규화된 좌표값이므로, 이미지 크기에 맞춰 변환 필요
- `cv2.circle()`을 사용해 각 점을 시각화 가능
- `refine_landmarks=True`로 눈/입 등 정밀도 향상 가능

---
### **📌 코드 설명**
### FaceMesh 얼굴 랜드마크 검출기 초기화

```python
with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as face_mesh:
```

- `refine_landmarks=True`: 눈동자, 입술 등 세부 영역까지 정밀 추적 가능  
- `max_num_faces=1`: 1명만 추적 (성능 고려)

---

### 정규화된 랜드마크를 픽셀 좌표로 변환 후 시각화

```python
h, w, _ = output_frame.shape
for lm in face_landmarks.landmark:
    x, y = int(lm.x * w), int(lm.y * h)
    cv2.circle(output_frame, (x, y), 1, (0, 255, 0), -1)
```

- Mediapipe의 `landmark.x`, `landmark.y`는 정규화 좌표 (0~1)  
- 이미지 크기 `(w, h)`와 곱해서 실제 픽셀 위치로 변환  
- 각 랜드마크를 초록색 점(`(0,255,0)`)으로 시각화

---

###  ESC 키 입력 시 종료

```python
if cv2.waitKey(1) & 0xFF == 27:
    break
```

- 실시간 영상 처리 중 **ESC 키(27)** 를 누르면 루프 종료  
- 일반적인 실시간 처리에서 종료 조건으로 자주 사용

---


## 핵심 개념 요약

| 항목 | 설명 |
|------|------|
| 입력 영상 | OpenCV를 통해 웹캠에서 실시간 캡처 |
| 얼굴 분석 | Mediapipe `FaceMesh`로 468개 얼굴 랜드마크 검출 |
| 시각화 | `cv2.circle()` 함수로 각 점을 초록색으로 표시 |
| 종료 | ESC 키 입력 시 프로그램 종료 |

---

#### 구현 결과

<img width="482" alt="image" src="https://github.com/user-attachments/assets/52c8e2ff-92f3-42d0-a288-9d44ca01a1c8" />


[HW6_02.py코드](https://github.com/daanhaa/Computer-Vision/blob/main/HW6_0416/HW6_02.py)
