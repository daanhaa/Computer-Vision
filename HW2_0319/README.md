# HW2
## 01 이진화 및 히스토그램 구하기
### **🔹 수행 과정** <br>
1️⃣ 이미지를 **그레이스케일로 변환**  <br>
2️⃣ 특정 임계값(Threshold)을 설정하여 **이진화(Binary Image) 적용**  <br>
3️⃣ **이진화된 이미지의 히스토그램을 계산하고 시각화**   <br>
<br>
### ✅ 구현 요구사항
💡 **OpenCV를 활용한 영상 처리 기법 적용**  
✔ `cv.imread()`를 사용하여 이미지를 불러오기 
✔ `cv.cvtColor()`를 이용하여 **그레이스케일 변환**  
✔ `cv.threshold()`를 활용하여 **이진화(Thresholding) 수행**  
✔ `cv.calcHist()`를 이용해 **히스토그램을 계산하고 시각화**  
✔ `matplotlib`을 사용하여 **히스토그램 그래프 출력** 
✔ **그레이스케일 이미지도 히스토그램 구하기**
<br>
### 🔎 구현 힌트
✔ `cv.threshold()` 함수의 **두 번째 인자**는 임계값이며, 기본적으로 `127` 사용  
✔ `plt.plot()`을 활용하여 **히스토그램을 시각적으로 확인** 가능  
✔ 히스토그램을 분석하면 **어두운 영역과 밝은 영역의 분포를 쉽게 파악** 가능

### **📌 코드 설명**

### **1️⃣ 이진화 및 히스토그램 구하기**
- 이미지를 불러와 **그레이스케일 변환 후 이진화 적용**  
- `cv.THRESH_OTSU`를 사용해 **자동 임계값 설정**  
- `cv.calcHist()`로 **히스토그램 계산 및 분석**  
- `matplotlib`을 사용해 **히스토그램 시각화**  
- `cv.imshow()`로 **이진화된 이미지 출력**  

<br>


### 구현 결과
![image](https://github.com/user-attachments/assets/7e83374c-5107-439f-a061-53ad4508279e)

### 🔗Github [HW2_01.py](https://github.com/daanhaa/Computer-Vision/blob/main/HW2_0319/HW2_01.py) 
---
<br>

## **02. 모폴로지 연산 적용하기**  

### **🔹 수행 과정**  
1️⃣ **이진화된 이미지에 모폴로지 연산 적용**  
2️⃣ **팽창(Dilation), 침식(Erosion), 열림(Open), 닫힘(Close) 연산 수행**  
3️⃣ **각 연산의 차이를 비교하고 원본 이미지와 함께 출력**  
<br>
### ✅ 구현 요구사항  
✔ `cv.getStructuringElement()`를 사용하여 **5×5 크기의 사각형 커널 생성**  
✔ `cv.morphologyEx()`를 이용해 각 **모폴로지 연산 적용**  
✔ `np.hstack()`을 활용하여 **결과 이미지를 한 줄로 정렬하여 출력**  
✔ `cv.imshow()`를 사용하여 **출력된 결과를 확인**  
<br>
### 🔎 구현 힌트  
✔ `cv.MORPH_DILATE`, `cv.MORPH_ERODE`, `cv.MORPH_OPEN`, `cv.MOxRPH_CLOSE` 연산  사용  
✔ 모폴로지 연산을 적용하면 **노이즈 제거 및 객체 경계선 보정이 가능**
✔ `cv.imshow()`를 활용해 **원본과 변환된 이미지를 한 화면에서 비교**
<br>
### **📌 코드 설명**
### **2️⃣ 모폴로지 연산 적용하기**
- **팽창(Dilation), 침식(Erosion), 열림(Opening), 닫힘(Closing) 연산 수행**  
- `cv.getStructuringElement()`로 **5×5 사각형 커널 생성**  
- `cv.morphologyEx()`로 **각 모폴로지 연산 적용**  
- `np.hstack()`으로 **모든 연산 결과를 하나의 이미지로 연결**  
- `cv.imshow()`로 **한 화면에서 비교 출력** 
<br>

### 구현 결과

![image](https://github.com/user-attachments/assets/ba4341a8-9d1c-4808-b071-be361f1cfae9)


### 🔗Github [HW2_02.py](https://github.com/daanhaa/Computer-Vision/blob/main/HW2_0319/HW2_02.py) 

---

## **03. 기하 연산 및 선형 보간 적용하기**  

### **🔹 수행 과정**  
1️⃣ **이미지를 45도 회전**  
2️⃣ **회전된 이미지를 1.5배 확대**  
3️⃣ **확대된 이미지에 선형 보간(Bilinear Interpolation) 적용**  
<br>
### ✅ 구현 요구사항  
✔ `cv.getRotationMatrix2D()`를 사용하여 **회전 변환 행렬 생성**  
✔ `cv.warpAffine()`을 이용하여 **이미지를 회전 및 확대**  
✔ `cv.INTER_LINEAR`을 사용하여 **선형 보간 적용**  
✔ 원본 이미지와 변환된 이미지를 **한 화면에서 비교하여 출력**  
<br>
### 🔎 구현 힌트  
✔ `cv.getRotationMatrix2D()`의 세 번째 인자는 **회전 중심 (cols/2, rows/2)**  
✔ `cv.warpAffine()`의 네 번째 인자는 **출력 이미지 크기 (int(cols*1.5), int(rows*1.5))**  
✔ `cv.INTER_LINEAR`을 사용하면 **확대 시에도 부드러운 결과를 얻을 수 있음**  
<br>
### **📌 코드 설명**
### **3️⃣ 기하 연산 및 선형 보간 적용하기**
- **이미지 45도 회전 후 1.5배 확대**  
- `cv.getRotationMatrix2D()`로 **회전 변환 행렬 생성**  
- `cv.warpAffine()`으로 **이미지 회전 및 확대 적용**  
- `cv.INTER_LINEAR`로 **확대 시 품질 유지**  
- `cv.imshow()`로 **원본과 변환된 이미지 비교 출력**  


### 구현 결과
![image](https://github.com/user-attachments/assets/0ed52bc8-2ecd-41d6-9cd0-0461cdc21323)

### 🔗Github [HW2_03.py](https://github.com/daanhaa/Computer-Vision/blob/main/HW2_0319/HW2_03.py) 

---
### ➕ 이미지 하나로 이어붙이기 & 이미지 사이즈 조절
- `np.hstack()`으로 **이진화 및 모폴로지 연산 결과 정렬**  
- `cv.resize()`로 **화면 크기에 맞게 자동 조정**  
- `cv.getWindowImageRect()`로 **화면 크기 유지**  
- `cv.INTER_AREA`로 **이미지 축소 시 품질 보정**  
- `cv.imshow()`로 **모든 결과를 한 화면에 출력**


---
