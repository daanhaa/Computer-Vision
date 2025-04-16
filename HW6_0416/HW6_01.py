import cv2 as cv
import numpy as np
from sort import Sort

# 클래스 이름 불러오기
with open("coco.names", "r") as f:
    class_names = [line.strip() for line in f.readlines()]

# YOLOv4 모델 로드
net = cv.dnn.readNet("yolov4.weights", "yolov4.cfg")
layer_names = net.getLayerNames()
out_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# SORT 추적기 초기화
tracker = Sort()

# 🎥 비디오 파일 열기 (slow_traffic_small.mp4)
cap = cv.VideoCapture("slow_traffic_small.mp4")
if not cap.isOpened():
    raise IOError("비디오 파일을 열 수 없습니다.")

colors = np.random.randint(0, 255, size=(100, 3))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]

    # YOLO 입력 전처리
    blob = cv.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(out_layers)

    boxes = []
    for output in outputs:
        for det in output:
            scores = det[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and class_names[class_id] == "car":  # 🚗 자동차만 추적
                cx, cy, bw, bh = (det[0:4] * np.array([w, h, w, h])).astype(int)
                x1 = int(cx - bw / 2)
                y1 = int(cy - bh / 2)
                x2 = x1 + bw
                y2 = y1 + bh
                boxes.append([x1, y1, x2, y2, confidence])

    # SORT에 박스 정보 전달
    dets = np.array(boxes)
    tracked = tracker.update(dets)

    # 추적 결과 시각화
    for d in tracked:
        x1, y1, x2, y2, obj_id = d.astype(int)
        color = colors[int(obj_id) % 100]
        cv.rectangle(frame, (x1, y1), (x2, y2), color.tolist(), 2)
        cv.putText(frame, f'ID: {int(obj_id)}', (x1, y1 - 10), cv.FONT_HERSHEY_SIMPLEX, 0.6, color.tolist(), 2)

    cv.imshow("YOLOv4 + SORT Tracking (Video)", frame)
    if cv.waitKey(1) & 0xFF == 27:  # ESC 키
        break

cap.release()
cv.destroyAllWindows()
