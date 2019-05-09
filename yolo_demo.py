import numpy as np
import cv2
import tensorflow as tf


class Yolov1(object):
    def __init__(self, weights_file):
        # yolo可以预测的物体（20类，如猫，狗，车等）
        self.classes_list = ["aeroplane", "bicycle", "bird", "boat", "bottle",
                        "bus", "car", "cat", "chair", "cow", "diningtable",
                        "dog", "horse", "motorbike", "person", "pottedplant",
                        "sheep", "sofa", "train", "tvmonitor"]
        # 预测图片的尺寸 448*448
        self.image_size = 448
        # 将图片分割成7*7的格子（注意不是真的分割）
        self.cell_size = 7
        # 一个格子的大小 448/7 = 64
        self.image_scale = self.image_size / self.cell_size
        # 每个格子预测2个box
        self.box_per_cell = 2
        self.classes = len(self.classes_list)
        # 一张图片最多的边框（这个参数是极大值抑制用的）及iou
        self.max_output_size = 10
        self.iou_threshold = 0.4
        # 预测分数的阈值，我这里设为0.1
        self.threshold = 0.1
        # 偏移坐标值 [7, 7 ,2]
        self.x_offset = np.transpose(np.reshape(
            np.array([np.arange(self.cell_size)] * self.cell_size * self.box_per_cell),
            [self.box_per_cell, self.cell_size, self.cell_size]),
            [1, 2, 0])
        self.y_offset = np.transpose(self.x_offset, [1, 0, 2])

        self.sess = tf.Session()
        # 网络结构
        self._build_net()
        # 解析预测值
        self._build_detector()
        # 解析权重
        self._load_weights(weights_file)

    def _build_net(self):
        print('Start to build the network ...')
        self.images = tf.placeholder(tf.float32, [None, self.image_size, self.image_size, 3])
        net = self._conv2d(self.images, 1, 64, 7, 2)
        net = self._maxpool(net, 1, 2, 2)

        net = self._conv2d(net, 2, 192, 3, 1)
        net = self._maxpool(net, 2, 2, 2)

        net = self._conv2d(net, 3, 128, 1, 1)

        net = self._conv2d(net, 4, 256, 3, 1)

        net = self._conv2d(net, 5, 256, 1, 1)

        net = self._conv2d(net, 6, 512, 3, 1)
        net = self._maxpool(net, 6, 2, 2)

        net = self._conv2d(net, 7, 256, 1, 1)

        net = self._conv2d(net, 8, 512, 3, 1)

        net = self._conv2d(net, 9, 256, 1, 1)

        net = self._conv2d(net, 10, 512, 3, 1)

        net = self._conv2d(net, 11, 256, 1, 1)

        net = self._conv2d(net, 12, 512, 3, 1)

        net = self._conv2d(net, 13, 256, 1, 1)

        net = self._conv2d(net, 14, 512, 3, 1)

        net = self._conv2d(net, 15, 512, 1, 1)

        net = self._conv2d(net, 16, 1024, 3, 1)
        net = self._maxpool(net, 16, 2, 2)

        net = self._conv2d(net, 17, 512, 1, 1)

        net = self._conv2d(net, 18, 1024, 3, 1)

        net = self._conv2d(net, 19, 512, 1, 1)

        net = self._conv2d(net, 20, 1024, 3, 1)

        net = self._conv2d(net, 21, 1024, 3, 1)

        net = self._conv2d(net, 22, 1024, 3, 2)

        net = self._conv2d(net, 23, 1024, 3, 1)

        net = self._conv2d(net, 24, 1024, 3, 1)
        net = self._flatten(net)

        net = self._fc(net, 25, 512, activation=tf.nn.leaky_relu)
        net = self._fc(net, 26, 4096, activation=tf.nn.leaky_relu)
        net = self._fc(net, 27, self.cell_size*self.cell_size*(self.classes + self.box_per_cell*5))
        self.predicts = net

    def _conv2d(self, x, id, num_filters, filter_size, stride):
        in_channels = x.get_shape().as_list()[-1]
        weight = tf.Variable(tf.truncated_normal([filter_size, filter_size, in_channels, num_filters], stddev=0.1))
        bias = tf.Variable(tf.zeros([num_filters, ]))
        # padding
        pad_size = filter_size // 2
        pad_mat = np.array([[0, 0], [pad_size, pad_size], [pad_size, pad_size], [0, 0]])
        x_pad = tf.pad(x, pad_mat)
        conv = tf.nn.conv2d(x_pad, weight, strides=[1, stride, stride, 1], padding='VALID')
        output = tf.nn.leaky_relu(tf.nn.bias_add(conv, bias), alpha=0.1)
        print("    Layer %d: type=Conv, num_filter=%d, filter_size=%d, stride=%d, output_shape=%s"
              % (id, num_filters, filter_size, stride, str(output.get_shape())))
        return output

    def _maxpool(self, x, id, pool_size, stride):
        output = tf.nn.max_pool(x, [1, pool_size, pool_size, 1],
                                strides=[1, stride, stride, 1], padding='SAME')
        print("    Layer %d: type=MaxPool, pool_size=%d, stride=%d, output_shape=%s"
              % (id, pool_size, stride, str(output.get_shape())))
        return output

    def _fc(self, x, id, num_out, activation=None):
        num_in = x.get_shape().as_list()[-1]
        weight = tf.Variable(tf.truncated_normal([num_in, num_out], stddev=0.1))
        bias = tf.Variable(tf.zeros([num_out, ]))
        output = tf.nn.xw_plus_b(x, weight, bias)
        if activation:
            output = activation(output, alpha=0.1)
        print("    Layer %d: type=Fc, num_out=%d, output_shape=%s"
              % (id, num_out, str(output.get_shape())))
        return output

    def _flatten(self, x):
        """flatten the x"""
        tran_x = tf.transpose(x, [0, 3, 1, 2])  # channle first mode
        nums = np.product(x.get_shape().as_list()[1:])
        return tf.reshape(tran_x, [-1, nums])

    def _build_detector(self):
        # 分界值
        index1 = self.cell_size*self.cell_size*self.classes
        index2 = index1 + self.cell_size*self.cell_size*self.box_per_cell

        # predicts [batch_size, 7*7*(20 + 2 + 2*4)
        # 分类预测值 [batch_size, 7, 7, 20]
        class_probs = tf.reshape(self.predicts[0, :index1], [self.cell_size, self.cell_size, self.classes])
        # 置信度 [batch_size, 7, 7, 2]
        confs = tf.reshape(self.predicts[0, index1:index2], [self.cell_size, self.cell_size, self.box_per_cell])
        # 预测边框 [batch_size, 7, 7, 2, 4]
        boxes = tf.reshape(self.predicts[0, index2:], [self.cell_size, self.cell_size, self.box_per_cell, 4])

        # 分类分数，转为[98, 20]分数值
        scores = tf.expand_dims(confs, axis=-1)*tf.expand_dims(class_probs, axis=2)
        scores = tf.reshape(scores, [-1, self.classes])

        """
        预测边框 [x, y, w, h]，x,y是box左上角坐标的偏移值, w, h是根号width, 根号height
        将其转为 [左上角， 右下角]
        """
        # 得到x, y坐标和真实长宽
        boxes = tf.stack([(boxes[:, :, :, 0] + tf.constant(self.x_offset, dtype=tf.float32)) * self.image_scale,
                          (boxes[:, :, :, 1] + tf.constant(self.y_offset, dtype=tf.float32)) * self.image_scale,
                          tf.square(boxes[:, :, :, 2]) * self.image_size,
                          tf.square(boxes[:, :, :, 3]) * self.image_size], axis=3)
        boxes = tf.reshape(boxes, [-1, 4])
        # 将其转为[左上角， 右下角]
        predict_boxes = tf.stack([boxes[:, 0] - boxes[:, 2]/2,
                                  boxes[:, 1] - boxes[:, 3]/2,
                                  boxes[:, 0] + boxes[:, 2]/2,
                                  boxes[:, 1] + boxes[:, 3]/2], axis=1)

        self.scores = scores
        self.boxes = predict_boxes

    def _load_weights(self, weights_file):
        print("Start to load weights from file:%s" % (weights_file))
        saver = tf.train.Saver()
        saver.restore(self.sess, weights_file)

    def detect_from_file(self, image_file):
        image = cv2.imread(image_file)
        img_resized = cv2.resize(image, (448, 448))
        img_RGB = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_resized_np = np.asarray(img_RGB)
        _images = np.zeros((1, 448, 448, 3), dtype=np.float32)
        _images[0] = (img_resized_np / 255.0) * 2.0 - 1.0
        scores, boxes = self.sess.run([self.scores, self.boxes], feed_dict={self.images: _images})
        scores, boxes, box_classes = self._pre_process(scores, boxes)
        return scores, boxes, box_classes

    def _pre_process(self, scores, boxes):
        # 得到每个boxes的分类
        box_classes = tf.argmax(scores, axis=1)
        # 得到每个boxes的最大分数
        box_class_scores = tf.reduce_max(scores, axis=1)
        # 剔除小于0.1阈值的数
        filter_mask = box_class_scores >= 0.1
        scores = tf.boolean_mask(box_class_scores, filter_mask)
        boxes = tf.boolean_mask(boxes, filter_mask)
        box_classes = tf.boolean_mask(box_classes, filter_mask)
        # 非极大值抑制
        nms_indices = tf.image.non_max_suppression(boxes, scores, 10, 0.4)
        boxes = tf.gather(boxes, nms_indices)
        scores = tf.gather(scores, nms_indices)
        box_classes = tf.gather(box_classes, nms_indices)

        scores = self.sess.run(scores)
        boxes = self.sess.run(boxes)
        box_classes = self.sess.run(box_classes)
        classes = []
        for i in box_classes:
            classes.append(self.classes_list[i])
        return scores, boxes, classes

if __name__ == '__main__':
    weights_path = r'./weights/YOLO_small.ckpt'
    image_path = r'./test_image/dog.jpg'
    predict = Yolov1(weights_path)
    scores, boxes, classes = predict.detect_from_file(image_path)
    image = cv2.imread(image_path)
    img_resized = cv2.resize(image, (448, 448))
    boxes = boxes.astype(np.int32)
    for i, box in enumerate(boxes):
        cv2.rectangle(img_resized, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
        cv2.putText(img_resized, classes[i] + ' : %.2f' % scores[i], (box[0] + 5, box[1] - 7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    cv2.imshow('test', img_resized)
    cv2.waitKey(0)
