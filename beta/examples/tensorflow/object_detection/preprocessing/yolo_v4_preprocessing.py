"""
 Copyright (c) 2020 Intel Corporation
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
      http://www.apache.org/licenses/LICENSE-2.0
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

import numpy as np
from PIL import Image

import tensorflow as tf
from beta.examples.tensorflow.common.object_detection.utils import box_utils
from beta.examples.tensorflow.common.object_detection.utils.yolo_v4_utils import normalize_image, letterbox_resize, random_resize_crop_pad, reshape_boxes, random_hsv_distort, random_horizontal_flip, random_vertical_flip, random_grayscale, random_brightness, random_chroma, random_contrast, random_sharpness, random_blur, random_rotate, random_mosaic_augment # random_motion_blur


class YOLOv4Preprocessor:
    """Parser to parse an image and its annotations into a dictionary of tensors."""
    def __init__(self, config, is_train):
        """Initializes parameters for parsing annotations in the dataset.
        """

        self._is_training = is_train

        # Data is parsed depending on the `is_training` flag
        if self._is_training:
            self._parse_fn = self._parse_train_data
            # self._annotation_file = 'train2017.txt'
        else:
            self._parse_fn = self._parse_train_data # self._parse_predict_data
            # self._annotation_file = 'val2017.txt'

        self._parse_fn2 = self._parse_train_data2

        # self._annotation_lines = self._get_annotation_lines()
        self.input_shape = config['input_shape']
        self.enhance_augment = config['enhance_augment']
        self.anchors = self.get_anchors(config['anchors_path'])
        self.num_classes = config['num_classes']
        self.multi_anchor_assign = config['multi_anchor_assign']
        # self.batch_size = config['global_batch_size']

    # def _get_annotation_lines(self):
    #     with open(self._annotation_file) as f:
    #         lines = f.readlines()
    #         lines = [line.strip() for line in lines]
    #
    #     lines_dict = {}
    #     for line in lines:
    #         line_data = line.split()
    #         if self._is_training:
    #             _, img_name = line_data[0].split('train2017/')
    #         else:
    #             _, img_name = line_data[0].split('val2017/')
    #         # lines_dict['image_name.jpg'] = boxes
    #         lines_dict[img_name] = np.array([np.array(list(map(int, box.split(',')))) for box in line_data[1:]])
    #
    #     # # KeyError: '000000015830.jpg'
    #     # print('len dict', len(lines_dict))
    #     # for i, key in enumerate(lines_dict):
    #     #     print(key, lines_dict[key])
    #     #     if i == 10:
    #     #         break
    #
    #     return lines_dict

    def get_anchors(self, anchors_path):
        '''loads the anchors from a file'''
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    def create_preprocess_input_fn(self):
        """Parses data to an image and associated training labels.
        """
        return self._parse_fn

    def create_preprocess_input_fn2(self):
        """Parses data to an image and associated training labels.
        """
        return self._parse_fn2

    def _parse_train_data(self, data):
        """Parses data for training and evaluation."""
        image = data['image'] # <dtype: 'uint8'> <class 'tensorflow.python.framework.ops.Tensor'> (None, None, 3)
        filename = data['source_filename']
        # print(data['source_id'])
        groundtruth_classes = data['groundtruth_classes']
        groundtruth_boxes = data['groundtruth_boxes']

        image, box = tf.py_function(self.preprocess, [image, filename, groundtruth_classes, groundtruth_boxes, self.input_shape], [tf.float64, tf.float64])
        image.set_shape([None, None, 3])
        box.set_shape([None, 5])

        out = {}
        out['image'] = image
        out['box'] = box
        out['filename'] = filename

        return out


    def preprocess(self, image, filename, groundtruth_classes, groundtruth_boxes, input_shape):

        image_np = image.numpy()
        # print('From inside TFDS:')
        # print('\nfilename', filename)
        # print('image init np\n', filename, image_np[100, 100, :])
        image_pil = Image.fromarray(image_np)
        # filename = filename.numpy().decode("utf-8")
        # print('\nfilename', filename)
        # boxes = self._annotation_lines[filename]
        # print('boxes', boxes)
        # print('groundtruth_classes', groundtruth_classes.numpy())
        # print('groundtruth_boxes', groundtruth_boxes.numpy())
        # print('image_np shape', image_np.shape)
        # filename 000000008010.jpg
        # boxes[[451 253 585 293 37]
        # [115 234 272 269 37]
        # [191 212 243 312 0]
        # [497 211 545 337 0]]
        # groundtruth_classes [37 37 0 0]
        # groundtruth_boxes [[0.5729345  0.7058125  0.66336346 0.9161875]
        # [0.52835214  0.18032813  0.60884875  0.42585936]
        # [0.48013544 0.2993906  0.7059368  0.38214064]
        # [0.47713318 0.77720314  0.7617833 0.85235935]]

        image_shape = tf.shape(input=image)[0:2]
        denormalized_boxes = box_utils.denormalize_boxes(groundtruth_boxes, image_shape)

        boxes = []
        for denormalized_box, category_id in zip(denormalized_boxes.numpy(), groundtruth_classes.numpy()):
            x_min = int(denormalized_box[1])
            y_min = int(denormalized_box[0])
            x_max = int(denormalized_box[3])
            y_max = int(denormalized_box[2])
            boxes.append([x_min, y_min, x_max, y_max, int(category_id)])
        boxes = np.array(boxes)
        # print('boxes_final', boxes)

        input_shape = input_shape.numpy()

        # print('data type in', type(image_pil))
        # print('\nimage init np(pil)\n', filename, np.array(image_pil)[100:102,100:102,:])

        image, box = self.get_ground_truth_data(image_pil, boxes, input_shape, filename)

        # print('\nimage preprocessed first stage\n', filename, image[100:102,100:102,:])

        image = tf.convert_to_tensor(image, dtype=tf.float64)
        box = tf.convert_to_tensor(box, dtype=tf.float64)

        return image, box

    def get_ground_truth_data(self, image, boxes, input_shape, filename, max_boxes=100):
        '''random preprocessing for real-time data augmentation'''
        # line = annotation_line.split()
        # image = Image.open(line[0]) # PIL Image object containing image data
        image_size = image.size
        model_input_size = tuple(reversed(input_shape))
        # boxes = np.array([np.array(list(map(int,box.split(',')))) for box in line[1:]])

        # random resize image and crop|padding to target size
        image, padding_size, padding_offset = random_resize_crop_pad(image, target_size=model_input_size)

        # print('\nafter random_resize_crop_pad\n', filename, np.array(image)[100:102,100:102,:])

        # random horizontal flip image
        image, horizontal_flip = random_horizontal_flip(image)

        # random adjust brightness
        image = random_brightness(image)

        # random adjust color level
        image = random_chroma(image)

        # random adjust contrast
        image = random_contrast(image)

        # random adjust sharpness
        image = random_sharpness(image)

        # random convert image to grayscale
        image = random_grayscale(image)

        # random do normal blur to image
        # image = random_blur(image)

        # random do motion blur to image
        # image = random_motion_blur(image, prob=0.2)

        # random vertical flip image
        image, vertical_flip = random_vertical_flip(image)

        # random distort image in HSV color space
        # NOTE: will cost more time for preprocess
        #       and slow down training speed
        # image = random_hsv_distort(image)

        # reshape boxes based on augment
        boxes = reshape_boxes(boxes, src_shape=image_size, target_shape=model_input_size,
                              padding_shape=padding_size, offset=padding_offset,
                              horizontal_flip=horizontal_flip,
                              vertical_flip=vertical_flip)

        # random rotate image and boxes
        image, boxes = random_rotate(image, boxes)

        if len(boxes) > max_boxes:
            boxes = boxes[:max_boxes]

        # prepare image & box data
        image_data = np.array(image)
        image_data = normalize_image(image_data)
        box_data = np.zeros((max_boxes, 5))
        if len(boxes) > 0:
            box_data[:len(boxes)] = boxes

        return image_data, box_data


    def _parse_train_data2(self, data):

        image_data = data['image']
        box_data = data['box']
        filename = data['filename']

        im_shape = image_data.shape
        image_data, out0, out1, out2 = tf.py_function(self.preprocess2, [image_data, box_data, filename], [tf.float64, tf.float32, tf.float32, tf.float32]) # , tf.float32
        image_data.set_shape(im_shape)
        out0.set_shape([im_shape[0], 19, 19, 3, 85])
        out1.set_shape([im_shape[0], 38, 38, 3, 85])
        out2.set_shape([im_shape[0], 76, 76, 3, 85])

        out = {}
        out['image_input'] = image_data
        out['y_true_0'] = out0
        out['y_true_1'] = out1
        out['y_true_2'] = out2
        out['filename'] = filename

        # return out, tf.zeros(64, dtype=tf.dtypes.float32)

        labels = {
            'y_true_0': out0,
            'y_true_1': out1,
            'y_true_2': out2
        }
        return image_data, labels


    def preprocess2(self, image_data, box_data, filename):
        image_data = image_data.numpy()
        box_data = box_data.numpy()

        # print('box_data[0] after batching', box_data[0])

        if self.enhance_augment == 'mosaic':
            # add random mosaic augment on batch ground truth data
            image_data, box_data = random_mosaic_augment(image_data, box_data, prob=0.2)

        y_true1, y_true2, y_true3 = self.preprocess_true_boxes(box_data, self.input_shape, self.anchors, self.num_classes, self.multi_anchor_assign)

        image_data = tf.convert_to_tensor(image_data, dtype=tf.float64)
        y_true1 = tf.convert_to_tensor(y_true1, dtype=tf.float32)
        y_true2 = tf.convert_to_tensor(y_true2, dtype=tf.float32)
        y_true3 = tf.convert_to_tensor(y_true3, dtype=tf.float32)

        # zeros = np.zeros(self.batch_size)
        # zeros = tf.convert_to_tensor(zeros, dtype=tf.float32)

        return image_data, y_true1, y_true2, y_true3

    def preprocess_true_boxes(self, true_boxes, input_shape, anchors, num_classes, multi_anchor_assign, iou_thresh=0.2):
        '''Preprocess true boxes to training input format

        Parameters
        ----------
        true_boxes: array, shape=(m, T, 5)
            Absolute x_min, y_min, x_max, y_max, class_id relative to input_shape.
        input_shape: array-like, hw, multiples of 32
        anchors: array, shape=(N, 2), wh
        num_classes: integer
        multi_anchor_assign: boolean, whether to use iou_thresh to assign multiple
                             anchors for a single ground truth

        Returns
        -------
        y_true: list of array, shape like yolo_outputs, xywh are reletive value

        '''
        assert (true_boxes[..., 4] < num_classes).all(), 'class id must be less than num_classes'
        num_layers = len(anchors)//3 # default setting
        anchor_mask = [[6,7,8], [3,4,5], [0,1,2]] if num_layers==3 else [[3,4,5], [0,1,2]]

        #Transform box info to (x_center, y_center, box_width, box_height, cls_id)
        #and image relative coordinate.
        true_boxes = np.array(true_boxes, dtype='float32')
        input_shape = np.array(input_shape, dtype='int32')
        boxes_xy = (true_boxes[..., 0:2] + true_boxes[..., 2:4]) // 2
        boxes_wh = true_boxes[..., 2:4] - true_boxes[..., 0:2]
        true_boxes[..., 0:2] = boxes_xy / input_shape[::-1]
        true_boxes[..., 2:4] = boxes_wh / input_shape[::-1]

        batch_size = true_boxes.shape[0]
        grid_shapes = [input_shape//{0:32, 1:16, 2:8}[l] for l in range(num_layers)]
        y_true = [np.zeros((batch_size, grid_shapes[l][0], grid_shapes[l][1], len(anchor_mask[l]), 5 + num_classes),
            dtype='float32') for l in range(num_layers)]

        # Expand dim to apply broadcasting.
        anchors = np.expand_dims(anchors, 0)
        anchor_maxes = anchors / 2.
        anchor_mins = -anchor_maxes
        valid_mask = boxes_wh[..., 0] > 0

        for b in range(batch_size):
            # Discard zero rows.
            wh = boxes_wh[b, valid_mask[b]]
            if len(wh) == 0:
                continue

            # Expand dim to apply broadcasting.
            wh = np.expand_dims(wh, -2)
            box_maxes = wh / 2.
            box_mins = -box_maxes

            intersect_mins = np.maximum(box_mins, anchor_mins)
            intersect_maxes = np.minimum(box_maxes, anchor_maxes)
            intersect_wh = np.maximum(intersect_maxes - intersect_mins, 0.)
            intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
            box_area = wh[..., 0] * wh[..., 1]
            anchor_area = anchors[..., 0] * anchors[..., 1]
            iou = intersect_area / (box_area + anchor_area - intersect_area)

            # Sort anchors according to IoU score
            # to find out best assignment
            best_anchors = np.argsort(iou, axis=-1)[..., ::-1]

            if not multi_anchor_assign:
                best_anchors = best_anchors[..., 0]
                # keep index dim for the loop in following
                best_anchors = np.expand_dims(best_anchors, -1)

            for t, row in enumerate(best_anchors):
                for l in range(num_layers):
                    for n in row:
                        # use different matching policy for single & multi anchor assign
                        if multi_anchor_assign:
                            matching_rule = (iou[t, n] > iou_thresh and n in anchor_mask[l])
                        else:
                            matching_rule = (n in anchor_mask[l])

                        if matching_rule:
                            i = np.floor(true_boxes[b, t, 0] * grid_shapes[l][1]).astype('int32')
                            j = np.floor(true_boxes[b, t, 1] * grid_shapes[l][0]).astype('int32')
                            k = anchor_mask[l].index(n)
                            c = true_boxes[b, t, 4].astype('int32')
                            y_true[l][b, j, i, k, 0:4] = true_boxes[b, t, 0:4]
                            y_true[l][b, j, i, k, 4] = 1
                            y_true[l][b, j, i, k, 5+c] = 1

        return y_true