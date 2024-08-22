import os
import cv2
import numpy
from onnxruntime import InferenceSession
import onnxruntime as ort
from hailo_platform import (HEF, Device, VDevice, HailoStreamInterface, InferVStreams, ConfigureParams,
    InputVStreamParams, OutputVStreamParams, InputVStreams, OutputVStreams, FormatType)

cwd = os.getcwd()

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
    return numpy.stack([x1, y1, x2, y2], axis=-1)


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
    return numpy.stack(outputs, axis=-1)


class FaceDetector:
    def __init__(self, onnx_path=None, session=None):
        self.session = session

        self.batched = False
        if self.session is None:
            assert onnx_path is not None
            assert os.path.exists(onnx_path)
            self.session = InferenceSession(
                "/home/rpi5/tapway/FaceDet-onnx/models_onnx/scrfd_2.5g_bn.onnx", providers=["CPUExecutionProvider"]
            )
            self.session.graph_optimization_level = (
                ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            )

        self.nms_thresh = 0.4
        self.center_cache = {}
        input_cfg = self.session.get_inputs()[0]
        print(input_cfg)
        # input_shape = input_cfg.shape
        input_shape = (1, 3, 640, 640)
        if isinstance(input_shape[2], str):
            self.input_size = None
        else:
            self.input_size = tuple(input_shape[2:4][::-1])
        input_name = input_cfg.name
        outputs = self.session.get_outputs()
        if len(outputs[0].shape) == 3:
            self.batched = True
        output_names = []
        for output in outputs:
            output_names.append(output.name)
        self.input_name = input_name
        self.output_names = output_names
        self.use_kps = False
        self._num_anchors = 1
        if len(outputs) == 6:
            self.fmc = 3
            self._feat_stride_fpn = [8, 16, 32]
            self._num_anchors = 2
        elif len(outputs) == 9:
            self.fmc = 3
            self._feat_stride_fpn = [8, 16, 32]
            self._num_anchors = 2
            self.use_kps = True
        elif len(outputs) == 10:
            self.fmc = 5
            self._feat_stride_fpn = [8, 16, 32, 64, 128]
            self._num_anchors = 1
        elif len(outputs) == 15:
            self.fmc = 5
            self._feat_stride_fpn = [8, 16, 32, 64, 128]
            self._num_anchors = 1
            self.use_kps = True

    def forward(self, x, score_thresh, hailo_inference_face=None):
        scores_list = []
        bboxes_list = []
        points_list = []
        input_size = tuple(x.shape[0:2][::-1])
        blob = cv2.dnn.blobFromImage(
            x, 1.0 / 128, input_size, (127.5, 127.5, 127.5), swapRB=True
        )
        # net_outs = self.session.run(self.output_names, {self.input_name: blob})
        # print(blob.shape)
        # hef = HEF("/home/rpi5/tapway/FaceDet-onnx/models_hailo/scrfd_2.5g.hef")

        # devices = Device.scan()

        # with VDevice(device_ids=devices) as target:
        #     configure_params = ConfigureParams.create_from_hef(hef, interface=HailoStreamInterface.PCIe)
        #     network_group = target.configure(hef, configure_params)[0]
        #     network_group_params = network_group.create_params()
        #     input_vstream_info = hef.get_input_vstream_infos()[0]
        #     output_vstream_info = hef.get_output_vstream_infos()[0]
        #     input_vstreams_params = InputVStreamParams.make_from_network_group(network_group, quantized=False, format_type=FormatType.FLOAT32)
        #     output_vstreams_params = OutputVStreamParams.make_from_network_group(network_group, quantized=False, format_type=FormatType.FLOAT32)
        #     height, width, channels = hef.get_input_vstream_infos()[0].shape
                
        #     with InferVStreams(network_group, input_vstreams_params, output_vstreams_params) as infer_pipeline:
        #         input_data = {input_vstream_info.name: blob}    
        #         with network_group.activate(network_group_params):
        #             net_outs = infer_pipeline.infer(input_data)
        net_outs = hailo_inference_face.run(blob.transpose((0, 2, 3, 1)))

        input_height = blob.shape[2]
        input_width = blob.shape[3]
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
                anchor_centers = numpy.stack(
                    numpy.mgrid[:height, :width][::-1], axis=-1
                )
                anchor_centers = anchor_centers.astype(numpy.float32)

                anchor_centers = (anchor_centers * stride).reshape((-1, 2))
                if self._num_anchors > 1:
                    anchor_centers = numpy.stack(
                        [anchor_centers] * self._num_anchors, axis=1
                    )
                    anchor_centers = anchor_centers.reshape((-1, 2))
                if len(self.center_cache) < 100:
                    self.center_cache[key] = anchor_centers

            pos_indices = numpy.where(scores >= score_thresh)[0]
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
        self, img, score_thresh=0.5, input_size=None, max_num=0, metric="default", hailo_inference_face=None
    ):
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
        resized_img = cv2.resize(img, (new_width, new_height))
        det_img = numpy.zeros((input_size[1], input_size[0], 3), dtype=numpy.uint8)
        det_img[:new_height, :new_width, :] = resized_img

        scores_list, bboxes_list, points_list = self.forward(det_img, score_thresh, hailo_inference_face)

        scores = numpy.vstack(scores_list)
        scores_ravel = scores.ravel()
        order = scores_ravel.argsort()[::-1]
        bboxes = numpy.vstack(bboxes_list) / det_scale
        if self.use_kps:
            points = numpy.vstack(points_list) / det_scale
        pre_det = numpy.hstack((bboxes, scores)).astype(numpy.float32, copy=False)
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
            offsets = numpy.vstack(
                [
                    (det[:, 0] + det[:, 2]) / 2 - img_center[1],
                    (det[:, 1] + det[:, 3]) / 2 - img_center[0],
                ]
            )
            offset_dist_squared = numpy.sum(numpy.power(offsets, 2.0), 0)
            if metric == "max":
                values = area
            else:
                values = (
                    area - offset_dist_squared * 2.0
                )  # some extra weight on the centering
            index = numpy.argsort(values)[::-1]  # some extra weight on the centering
            index = index[0:max_num]
            det = det[index, :]
            if points is not None:
                points = points[index, :]
        return det, points

    def nms(self, outputs):
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
            xx1 = numpy.maximum(x1[i], x1[order[1:]])
            yy1 = numpy.maximum(y1[i], y1[order[1:]])
            xx2 = numpy.minimum(x2[i], x2[order[1:]])
            yy2 = numpy.minimum(y2[i], y2[order[1:]])

            w = numpy.maximum(0.0, xx2 - xx1 + 1)
            h = numpy.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)

            indices = numpy.where(ovr <= thresh)[0]
            order = order[indices + 1]

        return keep


"""
after transpose only, no reshape included

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
            output[layer_from_shape[1, 80, 80, 8]].reshape(1, -1, 4),  # bbox 8
            output[layer_from_shape[1, 40, 40, 8]].reshape(1, -1, 4),  # bbox 16
            output[layer_from_shape[1, 20, 20, 8]].reshape(1, -1, 4),  # bbox 32
            output[layer_from_shape[1, 80, 80, 20]].reshape(1, -1, 10), # kps 8
            output[layer_from_shape[1, 40, 40, 20]].reshape(1, -1, 10), # kps 16
            output[layer_from_shape[1, 20, 20, 20]].reshape(1, -1, 10)  # kps 32
            ]
"""