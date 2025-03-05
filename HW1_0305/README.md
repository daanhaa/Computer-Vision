# HW1
## 이미지 불러오기 및 그레이스케일 변환
- OpenCV를사용하여 이미지를 불러오고 화면에 출력
- 원본 이미지와 그레이스케일로 변환된 이미지를 나란히 표시

- cv.imread()를사용하여 이미지 로드
- cv.cvtColor() 함수를 사용해 이미지를 그레이스케일로 변환
- np.hstack() 함수를 이용해 원본 이미지와 그레이스케일 이미지를 가로로 연결하여 출력
- cv.imshow()와 cv.waitKey()를 사용해 결과를 화면에 표시하고, 아무키나 누르면 창이 닫히도록 할 것.

### 코드 설명
- cv.imread(): 이미지를 불러오는 함수. BGR 형식으로 읽음  
- cv.cvtColor(): BGR 이미지를 그레이스케일로 변환  
- np.hstack(): 원본 이미지와 그레이스케일 이미지를 가로로 결합  
- cv.imshow(): 결합된 이미지를 화면에 출력  
- cv.waitKey(): 아무 키나 누르면 창이 닫힘  
- cv.destroyAllWindows(): 모든 창 닫기

### 결과 화면
![image](https://github.com/user-attachments/assets/5371a7d3-725d-4b4b-9fd8-e91951dbcf6e)

## 웹캠 영상에서 에지 검출
- 웹캠을 사용하여 실시간 비디오 스트림을 가져온다
- 각 프레임에서 Canny Edge Detection을 적용하여 에지를 검출하고 원본 영상과 함께 출력
- cv.VideoCapture()를 사용해 웹캠 영상을 로드
- 각프레임을그레이스케일로변환한후, cv.Canny() 함수를사용해에지검출수행
- 원본영상과에지검출영상을가로로연결하여화면에출력
- q 키를누르면영상창이종료

### 코드 설명
cv.VideoCapture(): 웹캠을 열어 실시간 비디오 스트림을 가져옴.
cv.cvtColor(): 실시간으로 받은 프레임을 그레이스케일로 변환.
cv.Canny(): 변환된 프레임에 Canny 에지 검출 적용.
np.hstack(): 원본 영상과 에지 검출된 영상을 가로로 결합.
cv.imshow(): 결합된 영상을 실시간으로 출력.
cv.waitKey(): q 키를 누르면 종료.
cap.release() & cv.destroyAllWindows(): 웹캠과 모든 창 닫기.


### 결과 화면
![image](https://github.com/user-attachments/assets/a5ca0608-07dc-42ee-82b3-bf74f815f420)



## 마우스로영역선택및ROI(관심영역) 추출
- 이미지를불러오고사용자가마우스로클릭하고드래그하여관심영역(ROI)을선택
- 선택한영역만따로저장하거나표시

- 이미지를불러오고화면에출력
- cv.setMouseCallback()을사용하여마우스이벤트를처리
- 사용자가클릭한시작점에서드래그하여사각형을그리며영역을선택
- 마우스를놓으면해당영역을잘라내서별도의창에출력
- r 키를누르면영역선택을리셋하고처음부터다시선택
- s 키를누르면선택한영역을이미지파일로저장

### 코드 설명
cv.setMouseCallback(): 마우스 클릭 및 드래그 이벤트 처리.
cv.rectangle(): 드래그 중인 사각형을 화면에 실시간 표시.
cv.imwrite(): 드래그로 선택한 ROI를 이미지 파일로 저장 (s 키).
r 키: 선택 초기화 → 원본 이미지로 복구.
cv.imshow(): 원본 이미지와 선택된 ROI를 표시.
cv.waitKey(): q 키를 누르면 종료.
cv.destroyAllWindows(): 모든 창 닫기.


### 결과 화면
![image](https://github.com/user-attachments/assets/09d29429-075d-45bb-afa5-54f078c4a43b)
