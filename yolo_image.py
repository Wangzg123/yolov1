from yolo import Yolov1
import cv2
import numpy as np

if __name__ == '__main__':
    weights_path = r'./weights/YOLO_small.ckpt'
    image_path = r'./test_image/dog.jpg'
    predict = Yolov1(weights_path)
    image = cv2.imread(image_path)
    h, w, _ = image.shape
    h_scale = h / 448
    w_scale = w / 448
    scores, boxes, classes = predict.detect_from_frame(image)
    print(scores.size)
    boxes[:, 0] = boxes[:, 0]*w_scale
    boxes[:, 1] = boxes[:, 1]*h_scale
    boxes[:, 2] = boxes[:, 2]*w_scale
    boxes[:, 3] = boxes[:, 3]*h_scale

    boxes = boxes.astype(np.int32)
    for i, box in enumerate(boxes):
        cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
        cv2.putText(image, classes[i] + ' : %.2f' % scores[i], (box[0] + 5, box[1] - 7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    cv2.imshow('test', image)
    cv2.waitKey(0)
