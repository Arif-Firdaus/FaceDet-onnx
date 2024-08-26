import os
import cv2
import numpy as np
from sys import platform
from multiprocessing import Process

cwd = os.getcwd()

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def distance2box(points, distance, max_shape=None):
    """
    Decode distance prediction to bounding box.

    Args:
        points (Tensor): Shape (n, 2), [x, y].
        distance (Tensor): Distance from the given point to 4
            boundaries (left, top, right, bottom).
        max_shape (tuple): Shape of the image.

    Returns:
        Tensor: Decoded bboxes.
    """
    x1 = points[:, 0] - distance[:, 0]
    y1 = points[:, 1] - distance[:, 1]
    x2 = points[:, 0] + distance[:, 2]
    y2 = points[:, 1] + distance[:, 3]
    if max_shape is not None:
        x1 = x1.clamp(min=0, max=max_shape[1])
        y1 = y1.clamp(min=0, max=max_shape[0])
        x2 = x2.clamp(min=0, max=max_shape[1])
        y2 = y2.clamp(min=0, max=max_shape[0])
    return np.stack([x1, y1, x2, y2], axis=-1)


def distance2kps(points, distance, max_shape=None):
    """
    Decode distance prediction to bounding box.

    Args:
        points (Tensor): Shape (n, 2), [x, y].
        distance (Tensor): Distance from the given point to 4
            boundaries (left, top, right, bottom).
        max_shape (tuple): Shape of the image.

    Returns:
        Tensor: Decoded bboxes.
    """
    outputs = []
    for i in range(0, distance.shape[1], 2):
        px = points[:, i % 2] + distance[:, i]
        py = points[:, i % 2 + 1] + distance[:, i + 1]
        if max_shape is not None:
            px = px.clamp(min=0, max=max_shape[1])
            py = py.clamp(min=0, max=max_shape[0])
        outputs.append(px)
        outputs.append(py)
    return np.stack(outputs, axis=-1)


class FaceDetector:
    """
    A class used to perform face detection using either an ONNX runtime session or HAILO Process.
    
    Attributes:
        session: ONNX runtime session or None if not using ONNX.
        input_name: The input layer name for the ONNX model.
        output_names: List of output layer names for the ONNX model.
        nms_thresh: Threshold for non-maximum suppression.
        center_cache: Cache for center points of anchors.
        input_size: The expected input size of the model.
        batched: Boolean indicating if inputs are batched.
        _num_anchors: Number of anchors used in the model.
        fmc: Number of feature map levels.
        _feat_stride_fpn: List of strides for feature pyramid networks.
        use_kps: Boolean indicating if keypoints are used.
    """

    def __init__(self, onnx_path=None, session=None):
        """
        Initializes the FaceDetector object with optional ONNX path or session.

        Args:
            onnx_path (str): Path to the ONNX model file.
            session: Existing ONNX runtime session.
        """
        self.session = session
        if onnx_path:
            from onnxruntime import InferenceSession
            import onnxruntime as ort
            if platform == "darwin":
                ep = ["CoreMLExecutionProvider", "CPUExecutionProvider"]
            else:
                ep = ["CPUExecutionProvider"]
            self.session = InferenceSession(
                onnx_path, providers=ep
            )
            self.session.graph_optimization_level = (
                ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            )
            input_cfg = self.session.get_inputs()[0]
            input_shape = input_cfg.shape
            input_name = input_cfg.name
            outputs = self.session.get_outputs()
            output_names = []
            for output in outputs:
                output_names.append(output.name)
            self.input_name = input_name
            self.output_names = output_names

        self.nms_thresh = 0.4
        self.center_cache = {}

        # Refer model architecture for these values
        input_shape = (1, 3, 640, 640)
        if isinstance(input_shape[2], str):
            self.input_size = None
        else:
            self.input_size = tuple(input_shape[2:4][::-1])
        self.batched = True
        self._num_anchors = 1
        self.fmc = 3
        self._feat_stride_fpn = [8, 16, 32]
        self._num_anchors = 2
        self.use_kps = False

    def forward(self, x, score_thresh, hailo_inference_face, network_group, input_vstreams_params, output_vstreams_params, input_vstream_info, result_queue):
        """
        Performs a forward pass of the network to obtain face detection results.

        Args:
            x: Input image data.
            score_thresh: Score threshold for detections.
            hailo_inference_face: Hailo inference function.
            network_group: Hailo network group.
            input_vstreams_params: Input stream parameters for Hailo.
            output_vstreams_params: Output stream parameters for Hailo.
            input_vstream_info: Input stream info for Hailo.
            result_queue: Queue to store inference results.

        Returns:
            scores_list: List of detection scores.
            bboxes_list: List of bounding boxes.
            points_list: List of keypoints.
        """
        scores_list = []
        bboxes_list = []
        points_list = []

        if not self.session:
            # Custom inference path using multiprocessing
            blob = np.expand_dims(x, axis=0)
            input_data = {input_vstream_info.name: blob}
            # Create infer process for Hailo
            infer_process = Process(target=hailo_inference_face, args=(network_group, input_vstreams_params, output_vstreams_params, input_data, result_queue, "face"))
            infer_process.start()
            _, output = result_queue.get()
            infer_process.join()

            # Mapping output shape to layer name
            layer_from_shape: dict = {output[key].shape:key for key in output.keys()}
            net_outs = [sigmoid(output[layer_from_shape[1, 80, 80, 2]].reshape(1, -1, 1)),  # score 8
                        sigmoid(output[layer_from_shape[1, 40, 40, 2]].reshape(1, -1, 1)),  # score 16
                        sigmoid(output[layer_from_shape[1, 20, 20, 2]].reshape(1, -1, 1)),  # score 32
                        output[layer_from_shape[1, 80, 80, 8]].reshape(1, -1, 4),           # bbox 8
                        output[layer_from_shape[1, 40, 40, 8]].reshape(1, -1, 4),           # bbox 16
                        output[layer_from_shape[1, 20, 20, 8]].reshape(1, -1, 4),           # bbox 32
                        output[layer_from_shape[1, 80, 80, 20]].reshape(1, -1, 10),         # kps 8
                        output[layer_from_shape[1, 40, 40, 20]].reshape(1, -1, 10),         # kps 16
                        output[layer_from_shape[1, 20, 20, 20]].reshape(1, -1, 10)          # kps 32
                        ]
        else:
            # ONNX inference path
            blob = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
            blob = np.expand_dims(blob, axis=0)
            net_outs = self.session.run(self.output_names, {self.input_name: blob.astype(np.float32).transpose(0, 3, 1, 2)/255.0})
            
        input_height = blob.shape[1]
        input_width = blob.shape[2]
        fmc = self.fmc
        for idx, stride in enumerate(self._feat_stride_fpn):
            if self.batched:
                scores = net_outs[idx][0]
                boxes = net_outs[idx + fmc][0]
                boxes = boxes * stride
                if self.use_kps:
                    points = net_outs[idx + fmc * 2][0] * stride
            else:
                scores = net_outs[idx]
                boxes = net_outs[idx + fmc]
                boxes = boxes * stride
                if self.use_kps:
                    points = net_outs[idx + fmc * 2] * stride

            height = input_height // stride
            width = input_width // stride
            key = (height, width, stride)
            if key in self.center_cache:
                anchor_centers = self.center_cache[key]
            else:
                anchor_centers = np.stack(
                    np.mgrid[:height, :width][::-1], axis=-1
                )
                anchor_centers = anchor_centers.astype(np.float32)

                anchor_centers = (anchor_centers * stride).reshape((-1, 2))
                if self._num_anchors > 1:
                    anchor_centers = np.stack(
                        [anchor_centers] * self._num_anchors, axis=1
                    )
                    anchor_centers = anchor_centers.reshape((-1, 2))
                if len(self.center_cache) < 100:
                    self.center_cache[key] = anchor_centers

            pos_indices = np.where(scores >= score_thresh)[0]
            bboxes = distance2box(anchor_centers, boxes)
            pos_scores = scores[pos_indices]
            pos_bboxes = bboxes[pos_indices]
            scores_list.append(pos_scores)
            bboxes_list.append(pos_bboxes)
            if self.use_kps:
                points = distance2kps(anchor_centers, points)
                points = points.reshape((points.shape[0], -1, 2))
                points_list.append(points[pos_indices])
        return scores_list, bboxes_list, points_list

    def detect(
        self, network_group=None, input_vstreams_params=None, output_vstreams_params=None, input_vstream_info=None, result_queue=None, 
        img=None, score_thresh=0.5, input_size=None, max_num=0, metric="default", hailo_inference_face=None
    ):
        """
        Performs face detection on an image.

        Args:
            network_group: Hailo network group.
            input_vstreams_params: Input stream parameters for Hailo.
            output_vstreams_params: Output stream parameters for Hailo.
            input_vstream_info: Input stream info for Hailo.
            result_queue: Queue to store inference results.
            img: The image in which to detect faces.
            score_thresh (float): Minimum score for valid detections.
            input_size: The size to which the input image should be resized.
            max_num (int): Maximum number of faces to detect.
            metric (str): Metric to sort detected faces.
            hailo_inference_face: Hailo inference function.

        Returns:
            det: Detected bounding boxes.
            points: Detected keypoints (if use_kps is True).
        """
        assert input_size is not None or self.input_size is not None
        input_size = self.input_size if input_size is None else input_size

        im_ratio = float(img.shape[0]) / img.shape[1]
        model_ratio = float(input_size[1]) / input_size[0]
        if im_ratio > model_ratio:
            new_height = input_size[1]
            new_width = int(new_height / im_ratio)
        else:
            new_width = input_size[0]
            new_height = int(new_width * im_ratio)
        det_scale = float(new_height) / img.shape[0]
        resized_img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
        det_img = np.zeros((input_size[1], input_size[0], 3), dtype=np.uint8)
        det_img[:new_height, :new_width, :] = resized_img

        scores_list, bboxes_list, points_list = self.forward(det_img, score_thresh, hailo_inference_face, 
                                                             network_group, input_vstreams_params, output_vstreams_params, input_vstream_info, result_queue)

        scores = np.vstack(scores_list)
        scores_ravel = scores.ravel()
        order = scores_ravel.argsort()[::-1]
        bboxes = np.vstack(bboxes_list) / det_scale
        if self.use_kps:
            points = np.vstack(points_list) / det_scale
        pre_det = np.hstack((bboxes, scores)).astype(np.float32, copy=False)
        pre_det = pre_det[order, :]
        keep = self.nms(pre_det)
        det = pre_det[keep, :]
        if self.use_kps:
            points = points[order, :, :]
            points = points[keep, :, :]
        else:
            points = None
        if 0 < max_num < det.shape[0]:
            area = (det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1])
            img_center = img.shape[0] // 2, img.shape[1] // 2
            offsets = np.vstack(
                [
                    (det[:, 0] + det[:, 2]) / 2 - img_center[1],
                    (det[:, 1] + det[:, 3]) / 2 - img_center[0],
                ]
            )
            offset_dist_squared = np.sum(np.power(offsets, 2.0), 0)
            if metric == "max":
                values = area
            else:
                values = (
                    area - offset_dist_squared * 2.0
                )  # some extra weight on the centering
            index = np.argsort(values)[::-1]  # some extra weight on the centering
            index = index[0:max_num]
            det = det[index, :]
            if points is not None:
                points = points[index, :]
        return det, points

    def nms(self, outputs):
        """
        Applies non-maximum suppression (NMS) to filter overlapping bounding boxes.

        Args:
            outputs: Array containing bounding boxes and their scores.

        Returns:
            keep: Indices of the bounding boxes to keep after NMS.
        """
        thresh = self.nms_thresh
        x1 = outputs[:, 0]
        y1 = outputs[:, 1]
        x2 = outputs[:, 2]
        y2 = outputs[:, 3]
        scores = outputs[:, 4]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)

            indices = np.where(ovr <= thresh)[0]
            order = order[indices + 1]

        return keep


"""
Output mapping of scrfd_2.5g.hef from Hailo model zoo. 
The output ends at transpose so we need to add reshape to each output and sigmoid.

{(1, 40, 40, 2): 'scrfd_2_5g/conv49',   score_16
(1, 20, 20, 2): 'scrfd_2_5g/conv55',    score_32
(1, 40, 40, 8): 'scrfd_2_5g/conv50',    bbox_16
(1, 80, 80, 2): 'scrfd_2_5g/conv42',    score_8
(1, 80, 80, 8): 'scrfd_2_5g/conv43',    bbox_8
(1, 80, 80, 20): 'scrfd_2_5g/conv44',   kps_8
(1, 20, 20, 20): 'scrfd_2_5g/conv57',   kps_32
(1, 20, 20, 8): 'scrfd_2_5g/conv56',    bbox_32
(1, 40, 40, 20): 'scrfd_2_5g/conv51'}   kps_16

endnodes = [infer_results[layer_from_shape[1, 80, 80, 4]],  # stride 8 
            infer_results[layer_from_shape[1, 80, 80, 1]],  # stride 8 
            infer_results[layer_from_shape[1, 80, 80, 80]], # stride 8 
            infer_results[layer_from_shape[1, 40, 40, 4]],  # stride 16
            infer_results[layer_from_shape[1, 40, 40, 1]],  # stride 16
            infer_results[layer_from_shape[1, 40, 40, 80]], # stride 16
            infer_results[layer_from_shape[1, 20, 20, 4]],  # stride 32
            infer_results[layer_from_shape[1, 20, 20, 1]],  # stride 32
            infer_results[layer_from_shape[1, 20, 20, 80]]  # stride 32
        ]

endnodes = [sigmoid(output[layer_from_shape[1, 80, 80, 2]].reshape(1, -1, 1)),  # score 8
            sigmoid(output[layer_from_shape[1, 40, 40, 2]].reshape(1, -1, 1)),  # score 16
            sigmoid(output[layer_from_shape[1, 20, 20, 2]].reshape(1, -1, 1)),  # score 32
            output[layer_from_shape[1, 80, 80, 8]].reshape(1, -1, 4),           # bbox 8
            output[layer_from_shape[1, 40, 40, 8]].reshape(1, -1, 4),           # bbox 16
            output[layer_from_shape[1, 20, 20, 8]].reshape(1, -1, 4),           # bbox 32
            output[layer_from_shape[1, 80, 80, 20]].reshape(1, -1, 10),         # kps 8
            output[layer_from_shape[1, 40, 40, 20]].reshape(1, -1, 10),         # kps 16
            output[layer_from_shape[1, 20, 20, 20]].reshape(1, -1, 10)          # kps 32
            ]
"""