import time
import os
import cv2
import numpy as np
from argparse import ArgumentParser
from multiprocessing import Process, Queue

from nets.nn import FaceDetector
from picamera2 import Picamera2
from libcamera import ColorSpace, controls
from hailo_platform import (
    HEF,
    VDevice,
    HailoStreamInterface,
    InferVStreams,
    ConfigureParams,
    InputVStreamParams,
    OutputVStreamParams,
    FormatType,
    HailoSchedulingAlgorithm,
)


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


def add_margin(
    img: np.ndarray,
    bbox: tuple,
    height_margin: float = 0.20,
    width_margin: float = 0.10,
) -> tuple:
    """
    Adjusts a bounding box by adding a margin around it.

    Args:
        img (numpy.ndarray): The image containing the bounding box.
        bbox (tuple): A tuple containing the bounding box coordinates (left, top, right, bottom).
        height_margin (float): The fractional margin to add to the height of the bounding box (default: 0.20).
        width_margin (float): The fractional margin to add to the width of the bounding box (default: 0.10).

    Returns:
        tuple: Adjusted bounding box coordinates (new_left, new_top, new_right, new_bottom) as integers,
               ensuring the bounding box stays within the image boundaries.
    """
    img_width, img_height = img.shape[1], img.shape[0]
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

    return int(new_left), int(new_top), int(new_right), int(new_bottom)


def preprocess_image(image: np.ndarray, input_size: tuple = (640, 640)) -> np.ndarray:
    """
    Preprocesses an image by resizing it to the specified input size and formatting it for model input.

    Args:
        image (numpy.ndarray): The original image to preprocess.
        input_size (tuple): The target size (width, height) for resizing the image (default: (640, 640)).

    Returns:
        numpy.ndarray: The preprocessed image, resized and expanded with an extra dimension to match model input requirements.
    """
    image_resized = cv2.resize(image, input_size, interpolation=cv2.INTER_LINEAR)
    image_resized = np.expand_dims(image_resized, axis=0)
    return image_resized.astype(np.uint8)


def hailo_session(hef: HEF, target: VDevice, output_dtype: str = "FLOAT32"):
    """
    Initializes and configures a Hailo session for model inference.

    Args:
        hef (HEF): The Hailo Execution File (HEF) that contains the model.
        target (VDevice): The Hailo VDevice target that manages the hardware resources.
        output_dtype (str): Data type of the output stream, default is "FLOAT32".

    Returns:
        tuple: Contains the network group, input vstreams parameters, output vstreams parameters, and input vstream info:
            - network_group (NetworkGroup): Configured network group for inference.
            - input_vstreams_params (InputVStreamParams): Parameters for the input virtual stream.
            - output_vstreams_params (OutputVStreamParams): Parameters for the output virtual stream.
            - input_vstream_info (VStreamInfo): Information about the input virtual stream.
    """
    configure_params = ConfigureParams.create_from_hef(
        hef=hef, interface=HailoStreamInterface.PCIe
    )
    network_groups = target.configure(hef, configure_params)
    network_group = network_groups[0]

    # Create input and output virtual streams params
    input_format_type = hef.get_input_vstream_infos()[0].format.type
    input_vstreams_params = InputVStreamParams.make_from_network_group(
        network_group, format_type=input_format_type
    )
    output_vstreams_params = OutputVStreamParams.make_from_network_group(
        network_group, format_type=getattr(FormatType, output_dtype)
    )

    # Define dataset params
    input_vstream_info = hef.get_input_vstream_infos()[0]

    return (
        network_group,
        input_vstreams_params,
        output_vstreams_params,
        input_vstream_info,
    )


def infer(
    network_group,
    input_vstreams_params,
    output_vstreams_params,
    input_data,
    infer_results_queue,
    queue_name: str,
):
    """
    Executes inference on a specified network group and places the results in a queue.

    Args:
        network_group (NetworkGroup): The configured network group to run the inference.
        input_vstreams_params (InputVStreamParams): Parameters for the input virtual stream.
        output_vstreams_params (OutputVStreamParams): Parameters for the output virtual stream.
        input_data (dict): The input data for inference, keyed by stream name.
        infer_results_queue (Queue): A multiprocessing queue to store the inference results.
        queue_name (str): The identifier for the queue entry (e.g., "face", "age", "gender").

    Returns:
        None. The inference results are put into the `infer_results_queue`.
    """
    with InferVStreams(
        network_group, input_vstreams_params, output_vstreams_params
    ) as infer_pipeline:
        infer_results = infer_pipeline.infer(input_data)
    infer_results_queue.put((queue_name, infer_results))


def main():
    """
    Main function to execute the face detection and age/gender classification pipeline.

    This function sets up the camera configuration, loads the Hailo models for face detection,
    age and gender classification, and processes video input either from a live camera feed or
    a pre-recorded video file. It performs inference using the Hailo device and displays the results.

    Args:
        None. All parameters are parsed from command-line arguments.

    Returns:
        None. This function runs indefinitely until manually terminated.
    """
    cwd = os.getcwd()
    parser = ArgumentParser()
    parser.add_argument(
        "--rtsp",
        dest="rtsp",
        action="store_true",
        help="use frames from rtsp stream for inference",
    )
    parser.add_argument(
        "--video-testing",
        dest="video_testing",
        action="store_true",
        help="write video stream with inference to demo folder",
    )
    parser.add_argument(
        "--margin",
        dest="margin",
        action="store_true",
        help="add margin to face bounding box",
    )
    args = parser.parse_args()

    # Initialize the face detector
    detector = FaceDetector()

    # * Define mappings for age and gender classification
    # Mapping from model output index to encoded age group
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
    # Mapping from encoded age group to age group string
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

    if not args.rtsp:
        # Picamera2 configuration setup
        #! Configuration options that affects the FPS by order (from highest to lowest):
        # ? noise reduction mode / Sensor modes / colour_space or format (memory too) / resolution / buffer_count / queue / HDR
        picam2 = Picamera2()
        picam2.set_controls(
            {
                "NoiseReductionMode": controls.draft.NoiseReductionModeEnum.Fast,  # ? HighQuality, Fast
                "HdrMode": controls.HdrModeEnum.Night,                             # ? Night, SingleExposure
            }
        )
        mode = picam2.sensor_modes[0]                      # ? There are three sensor modes available for the v3 camera / HDR only one
        camera_res_height = 960                            # ? Both resolution need to follow the given aligh_configuration resolution
        camera_res_width = 1536
        camera_config = picam2.create_video_configuration( # ? Configuration for main stream
            colour_space=ColorSpace.Rec709(),
            queue=True,
            sensor={"output_size": mode["size"], "bit_depth": mode["bit_depth"]},
            main={"size": (camera_res_width, camera_res_height), "format": "YUV420"},
            buffer_count=9,
        )                             
        picam2.align_configuration(camera_config)          # ? Align the configuration to the allowed values
        print(camera_config["main"])
        picam2.configure(camera_config)                    # ? Init the camera with the given configuration
        picam2.start()

    start_time = time.time()
    num_iterations = 0
    # Load compiled HEFs for face detection, age, and gender classification
    first_hef_path = cwd + "/models/hailo/scrfd_2.5g.hef"
    second_hef_path = cwd + "/models/hailo/age_F1.hef"
    third_hef_path = cwd + "/models/hailo/gender_F1.hef"
    first_hef = HEF(first_hef_path)
    second_hef = HEF(second_hef_path)
    third_hef = HEF(third_hef_path)
    hefs = [first_hef, second_hef, third_hef]

    # Creating the VDevice target with scheduler enabled (must for running multiple models in parallel)
    params = VDevice.create_params()
    params.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN
    params.multi_process_service = True
    with VDevice(params) as target:
        result_queue = Queue()

        # Create Hailo sessions for each model
        (
            face_network_group,
            face_input_vstreams_params,
            face_output_vstreams_params,
            face_input_vstream_info,
        ) = hailo_session(hefs[0], target)
        (
            age_network_group,
            age_input_vstreams_params,
            age_output_vstreams_params,
            age_input_vstream_info,
        ) = hailo_session(hefs[1], target)
        (
            gender_network_group,
            gender_input_vstreams_params,
            gender_output_vstreams_params,
            gender_input_vstream_info,
        ) = hailo_session(hefs[2], target)

        # Video output setup if video-testing is enabled
        if args.video_testing:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(
                cwd + "/demo/test_video_hailo_infer.mp4", fourcc, 15.0, (1920, 1080)
            )

            video_file = cwd + "/demo/test_video.mp4"
            cap = cv2.VideoCapture(video_file)
            if not cap.isOpened():
                print("Error: Could not open webcam.")
                return
        elif args.rtsp:
            # gst_pipeline = "rtspsrc protocols=udp location=rtsp://127.0.0.1:8554/cam1 latency=0 ! queue ! decodebin ! videoconvert ! video/x-raw,format=I420 ! appsink drop=1"
            gst_pipeline = 'rtspsrc protocols=tcp location="rtsp://admin:tapway123@192.168.0.198:554/cam/realmonitor?channel=1&subtype=1" latency=0 ! queue ! rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! video/x-raw,format=I420 ! appsink drop=1'
            
            cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
            if not cap.isOpened():
                print("Error: Could not open RTSP stream.")
                return
            print("Successfully connected to the RTSP stream.")

        # Main processing loop
        while True:
            if not args.video_testing and not args.rtsp:
                frame = picam2.capture_array("main")
                frame = cv2.cvtColor(frame, cv2.COLOR_YUV420p2RGB)
                frame = cv2.flip(frame, 1)
            else:
                ret, frame = cap.read()
                frame = cv2.cvtColor(frame, cv2.COLOR_YUV420p2RGB)
                if not ret:
                    print("Failed to grab a frame from the stream.")
                    break
            frame_height, frame_width, _ = frame.shape

            # Perform face detection
            boxes, points = detector.detect(
                face_network_group,
                face_input_vstreams_params,
                face_output_vstreams_params,
                face_input_vstream_info,
                result_queue,
                frame,
                score_thresh=0.62,
                input_size=(640, 640),
                hailo_inference_face=infer,
            )

            # Process each detected face
            for box in boxes:
                x1, y1, x2, y2, score = box

                if not args.margin:
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                else:
                    x1, y1, x2, y2 = add_margin(frame, (x1, y1, x2, y2))

                # Ensure the bounding box stays within the image boundaries
                x1, y1, x2, y2 = np.clip(
                    [x1, y1, x2, y2],
                    0,
                    [frame_width, frame_height, frame_width, frame_height],
                )
                cropped_image = frame[y1:y2, x1:x2]
                if cropped_image.shape[0] == 0 or cropped_image.shape[1] == 0:
                    continue
                processed_image = preprocess_image(cropped_image)

                # Create and start inference processes for age and gender classification
                input_data = {age_input_vstream_info.name: processed_image}
                age_infer_process = Process(
                    target=infer,
                    args=(
                        age_network_group,
                        age_input_vstreams_params,
                        age_output_vstreams_params,
                        input_data,
                        result_queue,
                        "age",
                    ),
                )
                age_infer_process.start()

                input_data = {gender_input_vstream_info.name: processed_image}
                gender_infer_process = Process(
                    target=infer,
                    args=(
                        gender_network_group,
                        gender_input_vstreams_params,
                        gender_output_vstreams_params,
                        input_data,
                        result_queue,
                        "gender",
                    ),
                )
                gender_infer_process.start()

                # Retrieve and process the inference results
                res_age, res_gender = None, None
                while not res_age or not res_gender:
                    model, res = result_queue.get()
                    if model == "age":
                        res_age = res
                        age_infer_process.join()
                    else:
                        res_gender = res
                        gender_infer_process.join()

                age = pred_to_age[age_to_age[res_age["age/softmax1"][0].argmax()]]
                gender = pred_to_gender[res_gender["gender/softmax1"][0].argmax()]

                # Annotate the frame with the age and gender predictions
                cv2.putText(
                    frame,
                    f"{gender}:{age}",
                    (x1, int(y1 * 0.99)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    2,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            if points is not None:
                for point in points:
                    for kp in point:
                        kp = kp.astype(np.int32)
                        cv2.circle(frame, tuple(kp), 1, (0, 255, 0), 2)

            iteration_time = time.time() - start_time
            num_iterations += 1
            fps = num_iterations / iteration_time
            frame = write_text_top_left(frame, f"{fps:.2f} FPS")
            if args.video_testing:
                out.write(frame)
            cv2.imshow("Webcam", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    # End of the main processing loop
    end_time = time.time()
    time_elapsed = end_time - start_time
    iterations_per_second = num_iterations / time_elapsed

    print(f"Total time elapsed: {time_elapsed} seconds")
    print(f"Iterations per second: {iterations_per_second}")
    if args.video_testing:
        cap.release()
        out.release()
    elif args.rtsp:
        cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
