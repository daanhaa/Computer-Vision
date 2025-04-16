import cv2 as cv
import mediapipe as mp

# MediaPipe 초기화
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

# 웹캠 열기
cap = cv.VideoCapture(0, cv.CAP_DSHOW)
if not cap.isOpened():
    print("웹캠을 열 수 없습니다.")
    exit()

with mp_face_mesh.FaceMesh(
        max_num_faces=2,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as face_mesh:

    while True:
        ret, frame = cap.read()
        if not ret:
            print("프레임 획득 실패")
            break

        # RGB로 변환 (MediaPipe는 RGB 필요)
        img_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

        # 얼굴 메시 처리
        results = face_mesh.process(img_rgb)

        # 결과 그리기
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_styles.get_default_face_mesh_tesselation_style())

        # 화면 출력
        cv.imshow("MediaPipe Face Mesh", cv.flip(frame, 1))  # 좌우반전

        # 종료 조건
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv.destroyAllWindows()
