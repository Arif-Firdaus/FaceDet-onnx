import os
import json
import numpy as np
import torch
import onnx
import onnxsim
from PIL import Image
from hailo_sdk_client import ClientRunner, __version__ as hailo_sdk_version
from datetime import datetime
import subprocess
import sys
import tensorflow as tf
from loguru import logger
from tqdm import tqdm
import platform
import psutil
from argparse import ArgumentParser

class EnvironmentSetup:
    """
    A class to handle the setup and verification of the environment required for model quantization and optimization.

    This class checks the installation of Hailo Dataflow Compiler, OS version, OS flavor, system RAM, Python version, ONNX version, onnxsim version, TensorFlow installation, CUDA version, cuDNN version, and NVIDIA DALI.
    It also installs NVIDIA DALI if it is not present in the environment.
    """

    def __init__(self):
        self.hailo_dataflow_compiler_version = None
        self.os_version = None
        self.ubuntu_version = None
        self.total_ram = None
        self.available_ram = None
        self.python_version = None
        self.onnx_version = None
        self.onnxsim_version = None
        self.tf_version = None
        self.gpu_devices = []
        self.cuda_version = None
        self.cudnn_version = None

    def check_hailo_dataflow_compiler_version(self):
        """
        Check the version of Hailo Dataflow Compiler installed in the environment.
        Logs Hailo Dataflow Compiler version.
        """
        try:
            self.hailo_dataflow_compiler_version = hailo_sdk_version
            logger.info(f"Hailo Dataflow Compiler version: {self.hailo_dataflow_compiler_version}")
        except Exception as e:
            logger.error(f"Failed to check Hailo Dataflow Compiler version: {e}")
            sys.exit(1)

    def check_os_version(self):
        """
        Check and log the OS version, including Ubuntu version if applicable.
        """
        try:
            self.os_version = platform.platform()
            if 'Linux' in self.os_version:
                try:
                    with open('/etc/os-release') as f:
                        for line in f:
                            if line.startswith('ID='):
                                distro_name = line.split('=')[1].strip().strip('"')
                            elif line.startswith('VERSION_ID='):
                                self.ubuntu_version = line.split('=')[1].strip().strip('"')

                    if distro_name == 'ubuntu':
                        logger.info(f"OS Version: {self.os_version}, Ubuntu Version: {self.ubuntu_version}")
                    else:
                        logger.info(f"OS Version: {self.os_version}, Distribution: {distro_name}")
                except FileNotFoundError:
                    logger.warning("Cannot determine Ubuntu version; /etc/os-release not found.")
                    logger.info(f"OS Version: {self.os_version}")
            else:
                logger.info(f"OS Version: {self.os_version}")
        except Exception as e:
            logger.error(f"Failed to check OS version: {e}")

    def check_system_ram(self):
        """
        Check and log the total and available system RAM.
        """
        try:
            self.total_ram = psutil.virtual_memory().total / (1024 ** 3)  # Convert to GB
            self.available_ram = psutil.virtual_memory().available / (1024 ** 3)  # Convert to GB
            logger.info(f"Total RAM: {self.total_ram:.2f} GB, Available RAM: {self.available_ram:.2f} GB")
        except Exception as e:
            logger.error(f"Failed to check system RAM: {e}")

    def check_python_version(self):
        """
        Check and log the Python version.
        """
        try:
            self.python_version = sys.version.split()[0]
            logger.info(f"Python Version: {self.python_version}")
        except Exception as e:
            logger.error(f"Failed to check Python version: {e}")

    def check_onnx_version(self):
        """
        Check and log the ONNX version.
        """
        try:
            self.onnx_version = onnx.__version__
            logger.info(f"ONNX Version: {self.onnx_version}")
        except Exception as e:
            logger.error(f"Failed to check ONNX version: {e}")
            sys.exit(1)

    def check_onnxsim_version(self):
        """
        Check and log the onnxsim version.
        """
        try:
            self.onnxsim_version = onnxsim.__version__
            logger.info(f"onnxsim Version: {self.onnxsim_version}")
        except Exception as e:
            logger.error(f"Failed to check onnxsim version: {e}")
            sys.exit(1)

    def check_tensorflow_installation(self):
        """
        Check if TensorFlow is installed and whether it can access the GPU. 
        Logs TensorFlow version and GPU availability.
        """
        try:
            self.tf_version = tf.__version__
            logger.info(f"TensorFlow version: {self.tf_version}")
        except ImportError:
            logger.error("TensorFlow is not installed.")
            sys.exit(1)

        # List available GPU devices
        self.gpu_devices = tf.config.list_physical_devices('GPU')
        if self.gpu_devices:
            logger.info(f"GPUs available: {[device.name for device in self.gpu_devices]}")
        else:
            logger.warning("No GPUs available. TensorFlow cannot access the GPU.")

    def check_cuda_and_cudnn_versions(self):
        """
        Check the versions of CUDA and cuDNN installed in the environment.
        Logs CUDA and cuDNN versions.
        """
        try:
            self.cuda_version = subprocess.check_output("nvcc --version", shell=True).decode('utf-8')
            logger.info(f"\nCUDA version: {self.cuda_version}")
        except Exception as e:
            logger.error(f"Failed to check CUDA version: {e}")
            sys.exit(1)

        try:
            self.cudnn_version = subprocess.check_output(
                "cat /usr/local/cuda/include/cudnn_version.h | grep CUDNN_MAJOR -A 2", shell=True
            ).decode('utf-8')
            logger.info(f"\ncuDNN version: {self.cudnn_version}")
        except Exception as e:
            logger.error(f"Failed to check cuDNN version: {e}")

    def check_and_install_dali(self):
        """
        Check if NVIDIA DALI is installed and install it if not found.
        This is used to improve data loading and processing performance.
        """
        try:
            import nvidia.dali as dali
            import nvidia.dali.plugin.tf as dali_tf
            logger.info("NVIDIA DALI and its TensorFlow plugin are already installed.")
        except ImportError:
            logger.warning("NVIDIA DALI or its TensorFlow plugin is not installed. Installing now...")
            self.install_dali()

    @staticmethod
    def install_dali():
        """
        Install NVIDIA DALI and its TensorFlow plugin from NVIDIA's repository.
        """
        try:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "nvidia-dali-cuda110", "--extra-index-url",
                 "https://developer.download.nvidia.com/compute/redist"]
            )
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "nvidia-dali-tf-plugin-cuda110", "--extra-index-url",
                 "https://developer.download.nvidia.com/compute/redist"]
            )
            logger.info("Successfully installed NVIDIA DALI and its TensorFlow plugin.")
        except subprocess.CalledProcessError as install_error:
            logger.error(f"Failed to install NVIDIA DALI or its TensorFlow plugin: {install_error}")
            sys.exit(1)

    def setup_versioning_directory(self):
        """
        Set up a directory for saving model versions and outputs, based on the current timestamp.
        This helps in organizing multiple runs and keeping track of different versions of model outputs.
        """
        base_directory = "runs"
        os.makedirs(base_directory, exist_ok=True)
        run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.version_directory = os.path.join(base_directory, f"run_{run_timestamp}")
        os.makedirs(self.version_directory, exist_ok=True)
        return self.version_directory

    def version_info(self):
        """
        Collect all version check information into a dictionary
        """
        
        version_info = {
            'version_directory': self.version_directory,
            'hailo_dataflow_compiler_version': self.hailo_dataflow_compiler_version,
            'os_version': self.os_version,
            'ubuntu_version': self.ubuntu_version,
            'total_ram_gb': self.total_ram,
            'available_ram_gb': self.available_ram,
            'python_version': self.python_version,
            'onnx_version': self.onnx_version,
            'onnxsim_version': self.onnxsim_version,
            'tensorflow_version': self.tf_version,
            'gpu_devices': self.gpu_devices,
            'cuda_version': self.cuda_version,
            'cudnn_version': self.cudnn_version
        }

        return version_info


class ModelManager:
    """
    A class to manage the model loading, exporting, optimization, and compilation process.

    This class provides functionalities to load models, export to ONNX, simplify models, translate to HAR format,
    optimize using Hailo SDK, compile the model to HEF, and run profiling commands.
    """

    def __init__(self, config, version_directory):
        """
        Initialize the ModelManager with configuration settings.
        
        Parameters:
            config (dict): Configuration settings loaded from a JSON file.
        """
        self.config = config
        self.model = None
        self.onnx_model = None
        self.version_directory = version_directory
        self.quantized_model_hef_path = None
        self.quantized_model_har_path = None
        self.runner = ClientRunner(hw_arch=config["hailo_hardware_arch"])


    def load_model(self):
        """
        Load the model based on the configuration file path.
        Supports loading both PyTorch (.pt) and ONNX (.onnx) model formats.
        """
        model_path = self.config["model_path"]
        if model_path.endswith(".pt"):
            self.model = torch.load(model_path)
        elif model_path.endswith(".onnx"):
            self.onnx_model = onnx.load(model_path)
        else:
            logger.error("Model path must end with '.pt' or '.onnx'.")
            sys.exit(1)

    def export_model_to_onnx(self):
        """
        Export a PyTorch model to ONNX format if a PyTorch model is loaded.
        The exported model is saved to the versioning directory.
        """
        if self.model is not None:
            input_tensor = torch.randn(
                (self.config["batch_size"], 3, *self.config["input_resolution"]), dtype=torch.float32
            )
            onnx_path = os.path.join(self.version_directory, f"{self.config['model_name']}.onnx")
            torch.onnx.export(
                self.model.eval(),
                input_tensor,
                onnx_path,
                verbose=False,
                input_names=self.config["input_names"],
                output_names=self.config["output_names"],
                opset_version=17,
            )
            self.onnx_model = onnx.load(onnx_path)

    def simplify_and_save_onnx_model(self):
        """
        Simplify the ONNX model to optimize it for deployment and save the simplified model.
        Simplification can reduce the model size and improve performance.
        """
        if self.onnx_model is not None:
            onnx.checker.check_model(self.onnx_model, full_check=True)
            model_simplified, check = onnxsim.simplify(self.onnx_model, check_n=3)
            # assert check, "Simplified ONNX model could not be validated successfully."
            if not check:
                logger.warning("Simplified ONNX model could not be validated successfully.")
            else:
                onnx_model_simplified_path = os.path.join(self.version_directory, f"{self.config['model_name']}_simplified.onnx")
                onnx.save(model_simplified, onnx_model_simplified_path)
                logger.info(f"Saved simplified ONNX to: {onnx_model_simplified_path}")

    def translate_to_har(self):
        """
        Translate the simplified ONNX model to a Hailo Archive File (HAR) using Hailo SDK.
        The HAR file is used for running models on Hailo hardware.
        """
        self.runner.translate_onnx_model(
            os.path.join(self.version_directory, f"{self.config['model_name']}_simplified.onnx"),
            self.config["model_name"],
            start_node_names=self.config["start_node_names"],
            end_node_names=self.config["end_node_names"],
        )
        logger.success("Hailo parsing completed (onnx -> HAR) - 3/6")
        self.runner.save_har(os.path.join(self.version_directory, f"{self.config['model_name']}.har"))

    def optimize_model(self, calib_dataset):
        """
        Optimize the model using Hailo SDK based on the calibration dataset.
        Saves the optimized model to a quantized HAR file.
        
        Parameters:
            calib_dataset (numpy.ndarray): The calibration dataset used for optimizing the model.
        
        Returns:
            str: The path to the quantized HAR file.
        """
        # Replace calibration sizes dynamically
        alls = [line.replace("num_calibration_images", str(calib_dataset.shape[0])) for line in self.config["alls"]]
        self.runner.load_model_script("".join(alls))
        self.runner.optimize(calib_dataset)
        logger.success("Hailo quantization and optimization completed (HAR fp32 -> HAR int8/int4) - 4/6")
        self.quantized_model_har_path = os.path.join(self.version_directory, f"{self.config['model_name']}_quantized.har")
        self.runner.save_har(self.quantized_model_har_path)

    def compile_model(self):
        """
        Compile the optimized model to a Hailo Executable File (HEF) which can be run on Hailo hardware.
        """
        hef = self.runner.compile()
        self.quantized_model_hef_path = os.path.join(self.version_directory, f"{self.config['model_name']}_quantized.hef")
        with open(self.quantized_model_hef_path, "wb") as f:
            f.write(hef)

    def run_profiler(self, version_directory):
        """
        Run the profiler command to evaluate the model's performance on Hailo hardware.
        
        Parameters:
            version_directory (str): The path to the run version directory to save logs.
        """
        profiler_html_output_path = os.path.join(version_directory, f"{self.config['model_name']}_quantized.html")
        profiler_csv_output_path = os.path.join(version_directory, f"{self.config['model_name']}_quantized.csv")
        run_shell_command(f"hailo profiler --hef {self.quantized_model_hef_path} --out-path {profiler_html_output_path} --csv {profiler_csv_output_path} {self.quantized_model_har_path}")

    def save_configuration(self, version_info):
        """
        Save the JSON configuration file used for this run to the versioning directory.
        This allows for reproducibility and understanding of the model settings used.
        """
        self.config['environment'] = version_info
        with open(os.path.join(self.version_directory, 'hailo_export_config.json'), 'w') as config_used_file:
            json.dump(self.config, config_used_file, indent=4)

    def run_cleanup(self):
        """
        Move all generated logs to the version directory and remove temp files
        """
        log_files = ['acceleras.log', 'allocator.log', 'hailo_sdk.client.log']
        for log in log_files:
            log_path = os.path.join(os.getcwd(), log)
            if os.path.exists(log_path):
                os.rename(log_path, os.path.join(self.version_directory, log))

        temp_files = ['estimation.csv', '*.client.log']
        for temp_file in temp_files:
            temp_file_path = os.path.join(os.getcwd(), temp_file)
            if os.path.exists(temp_file_path):
                os.remove(temp_file)


class DataPreprocessor:
    """
    A class to handle preprocessing of images for creating a calibration dataset.
    
    This class handles resizing and cropping images to the input size required by the model and
    creating a dataset for calibration purposes.
    """

    def __init__(self, config, version_directory):
        """
        Initialize the DataPreprocessor with configuration settings.
        
        Parameters:
            config (dict): Configuration settings loaded from a JSON file.
        """
        self.config = config
        self.calib_dataset = None
        self.version_directory = version_directory

    def preprocess_image(self, image, output_height=224, output_width=224, resize_side=224):
        """
        Efficiently resize and crop an image to the specified output size.

        Parameters:
            image (numpy.ndarray): The input image to preprocess.
            output_height (int): The height of the output image.
            output_width (int): The width of the output image.
            resize_side (int): The side length to which the smaller side of the image is resized before cropping.
        
        Returns:
            numpy.ndarray: The preprocessed image.
        """
        h, w = image.shape[:2]
        scale = min(resize_side / h, resize_side / w)
        new_height = int(h * scale)
        new_width = int(w * scale)

        # Convert image to TensorFlow tensor
        image = tf.convert_to_tensor(image)

        # Resize image
        resized_image = tf.image.resize(image, [new_height, new_width])

        # Central crop to the output size
        cropped_image = tf.image.resize_with_crop_or_pad(resized_image, output_height, output_width)

        return cropped_image

    def create_calibration_dataset(self):
        """
        Create or load a calibration dataset based on the configuration settings.
        If a dataset is provided, it loads it; otherwise, it preprocesses images from the specified directory.
        """
        images_path = self.config["images_path"]
        calib_data_npy = self.config.get("calib_data_npy")

        if calib_data_npy:
            self.calib_dataset = np.load(calib_data_npy)
        else:
            # List all images and preprocess them in batch
            images_list = [img_name for img_name in os.listdir(images_path) if img_name.lower().endswith(".jpg")]
            num_images = len(images_list)
            input_resolution = self.config["input_resolution"]
            self.calib_dataset = np.zeros((num_images, *input_resolution, 3), dtype=np.float32)

            # Step 2: Wrap the image processing loop with tqdm for a progress bar
            for idx, img_name in enumerate(tqdm(sorted(images_list), desc="Processing images")):
                img_path = os.path.join(images_path, img_name)
                img = np.array(Image.open(img_path).convert('RGB'))
                img_preproc = self.preprocess_image(img)
                self.calib_dataset[idx] = img_preproc.numpy()

            # Save the preprocessed dataset
            calib_data_path = os.path.join(self.version_directory, f"{self.config['model_name']}_{input_resolution[0]}_{input_resolution[1]}_calib_data.npy")
            np.save(calib_data_path, self.calib_dataset)
            logger.info(f"Saved calibration dataset to: {calib_data_path}")


def run_shell_command(command):
    """
    Run a shell command and return the output. Used for running system commands like the profiler.

    Parameters:
        command (str): The shell command to execute.

    Returns:
        str: The standard output from the command if it runs successfully; otherwise, exit.
    """
    try:
        result = subprocess.run(command, capture_output=True, text=True, shell=True, check=True)
        logger.info(f"\nCommand output: {result.stdout}")
        return result.stdout
    except subprocess.CalledProcessError as e:
        logger.error(f"\nCommand failed with error: {e.stderr}")
        sys.exit(1)
        

def main(config_file_path):
    """
    Main function to execute the script using OOP principles.
    This function orchestrates the overall workflow including environment setup, model management, 
    data preprocessing, optimization, and saving results.
    """
    # Load configuration from JSON file
    with open(config_file_path, 'r') as config_file:
        config = json.load(config_file)

    # Setup environment
    env_setup = EnvironmentSetup()
    version_directory = env_setup.setup_versioning_directory()

    # Configure logger
    if config['hailo_client_logging']:
        os.environ['HAILO_CLIENT_LOGS_ENABLED'] = "true"
        os.environ['HAILO_SDK_LOG_DIR'] = version_directory
    else:
        os.environ['HAILO_CLIENT_LOGS_ENABLED'] = "false"
    logger.add(os.path.join(version_directory, "loguru.log"), format="{time} {level} {message}", level="DEBUG", rotation="1 MB", retention="10 days")

    env_setup.check_hailo_dataflow_compiler_version()
    env_setup.check_os_version()
    env_setup.check_system_ram()
    env_setup.check_python_version()
    env_setup.check_onnx_version()
    env_setup.check_onnxsim_version()
    env_setup.check_tensorflow_installation()
    env_setup.check_cuda_and_cudnn_versions()
    env_setup.check_and_install_dali()
    version_info = env_setup.version_info()
    logger.success("Environment setup completed - 1/6")

    # Manage model operations
    model_manager = ModelManager(config, version_directory)
    model_manager.load_model()
    model_manager.export_model_to_onnx()
    model_manager.simplify_and_save_onnx_model()
    logger.success("ONNX export and simplify (pt -> onnx) completed - 2/6")
    model_manager.translate_to_har()

    # Data preprocessing
    data_preprocessor = DataPreprocessor(config, version_directory)
    data_preprocessor.create_calibration_dataset()

    # Optimize and compile model
    model_manager.optimize_model(data_preprocessor.calib_dataset)
    model_manager.compile_model()
    logger.success("Hailo compilation completed (HAR int8/int4 -> HEF) - 5/6")
    model_manager.run_profiler(version_directory)
    logger.success("Hailo profiling completed - 6/6")

    # Save configuration
    model_manager.save_configuration(version_info)

    # Cleanup
    model_manager.run_cleanup()

if __name__ == "__main__":
    parser = ArgumentParser(description='Export and optimize a model for Hailo hardware (Torch -> ONNX -> HAR -> HEF)')
    parser.add_argument('config_file_path', type=str, help='Path to the export JSON configuration file')
    args = parser.parse_args()
    main(args.config_file_path)

"""
python3 hailo_export.py configs/resnet18_age.json
"""