import time
import os
import cv2
import numpy as np
from sys import platform
from argparse import ArgumentParser

from nets.nn import FaceDetector
from onnxruntime import InferenceSession
import onnxruntime as ort


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
    image_resized = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
    image_resized = image_resized / 255.0
    image_resized = np.transpose(image_resized, (2, 0, 1))
    image_resized = np.expand_dims(image_resized, axis=0)
    return image_resized.astype(np.float32)


def main():
    """
    Main function to execute the face detection and age/gender classification pipeline.

    This function sets up the camera configuration, loads the ONNX models for face detection,
    age and gender classification, and processes video input from a live camera feed.
    It performs inference using the onnxruntime and displays the results.

    Args:
        None. All parameters are parsed from command-line arguments.

    Returns:
        None. This function runs indefinitely until manually terminated.
    """
    cwd = os.getcwd()
    parser = ArgumentParser()
    parser.add_argument(
        "--face-model",
        type=str,
        default=cwd + "/models/onnx/scrfd_2.5g_bn.onnx",
        help="model file path",
    )
    parser.add_argument(
        "--write-video",
        dest="write_video",
        action="store_true",
        help="write video stream with/without inference to demo folder",
    )
    parser.add_argument(
        "--margin",
        dest="margin",
        action="store_true",
        help="add margin to face bounding box",
    )
    args = parser.parse_args()

    # Initialize the face detector
    detector = FaceDetector(args.face_model)

    if platform == "darwin":
        ep = ["CoreMLExecutionProvider", "CPUExecutionProvider"]
    else:
        ep = ["CPUExecutionProvider"]

    # ONNX model initialization
    age_session = InferenceSession(
        cwd + "/models/onnx/yolov8n_age_train.onnx",
        providers=ep,
    )
    age_session.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    gender_session = InferenceSession(
        cwd + "/models/onnx/yolov8n_gender_train.onnx",
        providers=ep,
    )
    gender_session.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

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

    cap = cv2.VideoCapture(0)
    # Video output setup if write-video is enabled
    if args.write_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter("demo/test_video.mp4", fourcc, 15.0, (1920, 1080))
        out_infer = cv2.VideoWriter(
            "demo/test_video_onnx_infer.mp4", fourcc, 15.0, (1920, 1080)
        )
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    start_time = time.time()
    num_iterations = 0

    # Main processing loop
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_height, frame_width, _ = frame.shape
        frame = cv2.flip(frame, 1)
        if args.write_video:
            out.write(frame)

        # Perform face detection
        boxes, points = detector.detect(
            img=frame, score_thresh=0.5, input_size=(640, 640)
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

            # Perform age and gender inference
            res_age = age_session.run(None, {"images": processed_image})
            res_gender = gender_session.run(None, {"images": processed_image})
            age = pred_to_age[age_to_age[res_age[0].argmax()]]
            gender = pred_to_gender[res_gender[0].argmax()]

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

        iteration_time = time.time() - start_time
        num_iterations += 1
        fps = num_iterations / iteration_time
        frame = write_text_top_left(frame, f"{fps:.2f} FPS")
        if args.write_video:
            out_infer.write(frame)
        cv2.imshow("Webcam", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # End of the main processing loop
    end_time = time.time()
    time_elapsed = end_time - start_time
    iterations_per_second = num_iterations / time_elapsed

    print(f"Total time elapsed: {time_elapsed} seconds")
    print(f"Iterations per second: {iterations_per_second}")
    cap.release()
    if args.write_video:
        out.release()
        out_infer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
