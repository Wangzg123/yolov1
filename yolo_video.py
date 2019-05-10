import cv2
import numpy as np
from yolo import Yolov1

weights_path = r'./weights/YOLO_small.ckpt'
video_path = r'./test_image/road_video_compressed.mp4'
predict = Yolov1(weights_path)
cameraCapture = cv2.VideoCapture(video_path)
cv2.namedWindow('Test camera')
success = False
i = 0
if cameraCapture.isOpened():  # 判断是否正常打开
    success = True
while success:
    # # 27 == esc buttom
    if cv2.waitKey(1) == 27:
        break
    success, frame = cameraCapture.read()
    if success:
        h, w, _ = frame.shape
        print('[{}] width = {}, height = {}'.format(i, w, h))
        i = i + 1
        h_scale = h / 448
        w_scale = w / 448
        scores, boxes, classes = predict.detect_from_frame(frame)
        if scores.size:
            boxes[:, 0] = boxes[:, 0] * w_scale
            boxes[:, 1] = boxes[:, 1] * h_scale
            boxes[:, 2] = boxes[:, 2] * w_scale
            boxes[:, 3] = boxes[:, 3] * h_scale
            boxes = boxes.astype(np.int32)
            for i, box in enumerate(boxes):
                cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                cv2.putText(frame, classes[i] + ' : %.2f' % scores[i], (box[0] + 5, box[1] - 7),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.imshow('Test camera', frame)
    else:
        print('read mp4 failed')

cv2.destroyAllWindows()
cameraCapture.release()
