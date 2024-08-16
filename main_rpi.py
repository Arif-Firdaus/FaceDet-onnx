import time
import os
import warnings
from argparse import ArgumentParser

import cv2
import numpy as np

from nets.nn import FaceDetector

from picamera2 import Picamera2
from libcamera import ColorSpace, controls

# from multiprocessing import Process
# from hailo_platform import (HEF, Device, VDevice, HailoStreamInterface, InferVStreams, ConfigureParams,
#     InputVStreamParams, OutputVStreamParams, InputVStreams, OutputVStreams, FormatType)
from utils import HailoAsyncInference

# warnings.filterwarnings("ignore")

mean = [0.0, 0.0, 0.0]
std = [1.0, 1.0, 1.0]


def write_text_top_left(image, text, font_scale=1.5, color=(0, 255, 0), thickness=3):
    """
    Writes text at the top left corner of an image using OpenCV.

    Args:
        image: The image to write text on (numpy array).
        text: The text to write.
        font_scale: The font size scaling factor (default: 1.0).
        color: The color of the text in BGR format (default: white).
        thickness: The thickness of the text lines (default: 1).

    Returns:
        The image with text written on it (numpy array).
    """

    # Get image dimensions
    height, width, channels = image.shape

    # Estimate text size using cv2.getTextSize
    (text_width, text_height), _ = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_DUPLEX, font_scale, thickness
    )

    # Calculate top left corner coordinates for the text
    text_x = 10  # Adjust horizontal offset as needed (default: 10px)
    text_y = (
        text_height + 10
    )  # Adjust vertical offset as needed (default: add text height + 10px margin)

    # Ensure text stays within image boundaries
    text_y = min(text_y, height - 10)  # Limit text_y to avoid going below the image

    # Add the text using cv2.putText
    cv2.putText(
        image,
        text,
        (text_x, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        color,
        thickness,
    )

    return image


def add_margin(img, bbox, height_margin=0.20, width_margin=0.10):
    img_width, img_height = img.shape[1], img.shape[0]
    # Unpack the bounding box coordinates
    left, top, right, bottom = bbox

    # Calculate the width and height of the current bounding box
    width = right - left
    height = bottom - top

    # Calculate the margin
    margin_x = width * width_margin
    margin_y = height * height_margin
    margin_y_bottom = height * 0.05

    # Adjust the bounding box to include the margin
    new_left = max(0, left - margin_x)
    new_top = max(0, top - margin_y)
    new_right = min(img_width, right + margin_x)
    new_bottom = min(img_height, bottom + margin_y_bottom)
    # new_bottom = bottom

    return int(new_left), int(new_top), int(new_right), int(new_bottom)


def preprocess_image(image, input_size=(640, 640)):
    image_resized = cv2.resize(image, input_size, interpolation=cv2.INTER_LINEAR)
    image_resized = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
    image_resized = (image_resized)
    # image_resized = (image_resized / 255.0 - mean) / std
    # image_resized = np.transpose(image_resized, (2, 0, 1))
    image_resized = np.expand_dims(image_resized, axis=0)
    return image_resized.astype(np.float32)


def main():
    cwd = os.getcwd()
    parser = ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="/home/rpi5/tapway/FaceDet-onnx/models_hailo/scrfd_2.5g.hef",
        help="model file path",
    )

    args = parser.parse_args()
    detector = FaceDetector(args.model)

    # hef = HEF(cwd + "/models_hailo/age.hef")
    # hef = HEF("/home/rpi5/tapway/FaceDet-onnx/models_hailo/age.hef")

    # age_session = InferenceSession(
    #     cwd + "/models_onnx/yolov8n_age_train.onnx",
    #     providers=["CPUExecutionProvider"],
    # )
    # age_session.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    # gender_session = InferenceSession(
    #     cwd + "/models_onnx/yolov8n_gender_train.onnx",
    #     providers=["CPUExecutionProvider"],
    # )
    # gender_session.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    age_to_age = {
        0: 0,
        1: 1,
        2: 10,
        3: 11,
        4: 12,
        5: 13,
        6: 14,
        7: 2,
        8: 3,
        9: 4,
        10: 5,
        11: 6,
        12: 7,
        13: 8,
        14: 9,
    }

    pred_to_age = {
        0: "0-4",
        1: "5-9",
        2: "10-14",
        3: "15-19",
        4: "20-24",
        5: "25-29",
        6: "30-34",
        7: "35-39",
        8: "40-44",
        9: "45-49",
        10: "50-54",
        11: "55-59",
        12: "60-64",
        13: "65-69",
        14: ">70",
    }
    pred_to_gender = {0: "F", 1: "M"}

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
    hailo_inference = HailoAsyncInference(cwd + "/models_hailo/age.hef", 1)
    hailo_inference_face = HailoAsyncInference(cwd + "/models_hailo/scrfd_2.5g.hef", 1)
    height, width, _ = hailo_inference.get_input_shape()

    picam2 = Picamera2()
    picam2.set_controls(
        {"NoiseReductionMode": controls.draft.NoiseReductionModeEnum.HighQuality}
    )
    mode = picam2.sensor_modes[1]
    resolution = 320 * 3
    camera_config = picam2.create_video_configuration(
        colour_space=ColorSpace.Rec709(),
        queue=True,
        sensor={"output_size": mode["size"], "bit_depth": mode["bit_depth"]},
        main={"size": (resolution, resolution), "format": "YUV420"},
        buffer_count=3,
    )
    # ? FPS
    # ? Sensor modes
    # ? noise reduction
    # ? CMA memory
    picam2.align_configuration(camera_config)
    print(camera_config["main"])
    picam2.configure(camera_config)
    picam2.start()

    start_time = time.time()
    num_iterations = 0
    while True:
        frame = picam2.capture_array("main")
        frame = cv2.cvtColor(frame, cv2.COLOR_YUV420p2BGR)
        frame = cv2.flip(frame, 1)

        boxes, points = detector.detect(frame, score_thresh=0.5, input_size=(640, 640), hailo_inference_face)
        for box in boxes:
            x1, y1, x2, y2, score = box
            x1, y1, x2, y2 = add_margin(frame, (x1, y1, x2, y2))
            cropped_image = frame[y1:y2, x1:x2]
            processed_image = preprocess_image(cropped_image)
            # with InferVStreams(network_group, input_vstreams_params, output_vstreams_params) as infer_pipeline:
            #     input_data = {input_vstream_info.name: processed_image}    
            #     with network_group.activate(network_group_params):
            #         res_age = infer_pipeline.infer(input_data)

            res_age = hailo_inference.run(processed_image)

            # res_age = age_session.run(None, {"images": processed_image})
            # res_gender = gender_session.run(None, {"images": processed_image})
            # print(res_age["age/softmax1"])
            age = pred_to_age[age_to_age[res_age["age/softmax1"][0].argmax()]]
            # gender = pred_to_gender[res_gender[0].argmax()]
            cv2.putText(
                frame,
                f"{age}",
                # f"{gender}:{age}",
                (x1, int(y1 * 0.99)),
                cv2.FONT_HERSHEY_SIMPLEX,
                2,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # if points is not None:
        #     for point in points:
        #         for kp in point:
        #             kp = kp.astype(np.int32)
        #             cv2.circle(frame, tuple(kp), 1, (0, 255, 0), 2)
        iteration_time = time.time() - start_time
        num_iterations += 1
        fps = num_iterations / iteration_time
        frame = write_text_top_left(frame, f"{fps:.2f} FPS")
        cv2.imshow("Webcam", cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    end_time = time.time()
    time_elapsed = end_time - start_time
    iterations_per_second = num_iterations / time_elapsed

    print(f"Total time elapsed: {time_elapsed} seconds")
    print(f"Iterations per second: {iterations_per_second}")
    cv2.destroyAllWindows()
    hailo_inference.release_device()


if __name__ == "__main__":
    main()
