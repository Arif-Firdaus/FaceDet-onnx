# # Step 1: Use Ubuntu 22.04 as the base image
# FROM --platform=linux/amd64 ubuntu:22.04

# # Step 2: Set the Python version argument and non-interactive environment
# ARG PYTHON_VERSION=3.8
# ENV DEBIAN_FRONTEND=noninteractive

# # Step 3: Update and install necessary base packages
# RUN --mount=type=cache,target=/root/.cache/apt apt update && \
#     apt upgrade -y && \
#     apt install -y software-properties-common tzdata && \
#     ln -fs /usr/share/zoneinfo/Etc/UTC /etc/localtime && \
#     dpkg-reconfigure --frontend noninteractive tzdata && \
#     add-apt-repository -y ppa:deadsnakes/ppa && \
#     apt update && \
#     apt install -y \
#         python${PYTHON_VERSION} \
#         python${PYTHON_VERSION}-distutils \
#         python${PYTHON_VERSION}-dev \
#         wget \
#         gnupg \
#         graphviz \
#         libgraphviz-dev

# # Step 4: Install pip for Python
# RUN --mount=type=cache,target=/root/.cache/pip wget https://bootstrap.pypa.io/get-pip.py && \
#     python${PYTHON_VERSION} get-pip.py && \
#     rm get-pip.py

# # Step 5: Install CUDA 12.0
# RUN --mount=type=cache,target=/root/.cache/apt \
#     apt-key del 7fa2af80 && \
#     wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb && \
#     dpkg -i cuda-keyring_1.0-1_all.deb && \
#     apt update && \
#     apt install -y cuda-12-0 && \
#     rm -rf cuda-keyring_1.0-1_all.deb

# # Step 6: Install cuDNN compatible with CUDA 12.0
# # RUN --mount=type=cache,target=/root/.cache/apt apt install -y zlib1g && \
# #     wget -O "cudnn-local-repo-ubuntu2204-8.9.7.29_1.0-1_amd64.deb" "https://tapway-nvidia.s3.ap-southeast-1.amazonaws.com/cudnn-local-repo-ubuntu2204-8.9.7.29_1.0-1_amd64.deb?response-content-disposition=inline&X-Amz-Security-Token=IQoJb3JpZ2luX2VjELL%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaDmFwLXNvdXRoZWFzdC0xIkYwRAIgQ%2BeOID5oX3%2FGfeTfnCcZG4prmOESkOxhDamdZAkdp%2FQCIGaBqYhunWzQ1uc39IEs79kH69elHQiW3LS4%2Bf2NsSiYKu0CCMv%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEQAhoMNzc1Mjk4NDcwMjYzIgzx%2FV9OdivFcousCTEqwQLQJ%2FY5FLHWToF2be0oZ0TaqlPPLxSYuIb0e9CAMHJpemvPB%2FgCF0Dqj3ojnW7rjXRPrlKHeQGCau%2FcApdW8IgFB3qZN%2F3QmsB0ZMenVNE3sSlEKpzdeQKyXfqOsb1SuI9rB1B6OSnmyu26JhLcURe%2FzTxXHUetOXYF1dF%2F1oyw4c2l%2FDqfmB3g8qW4s2r3GKRxI%2F7iv7EeSQm8UOSZswqftfN8e02Vt0wrCaqDVNlol2Hjvl6A0N%2FH6ArrjWrpAzc%2BjvjzVvFhev9tJLadNvbooMh1lnITQO0LmbkKyxN3nxpJVBag%2BG3QrYA%2FF4%2BXG3o8Y7I3MPCzMfxso4TLPx9aZSbMxAJcibg0kX1XSRT9%2BiBk%2BLF3QeODoVNSAY84NV3pZ45nDLSOUtYVrHBMlyu2XhuMssltVyZZiaZWBSqo2wAwkP3etgY6tAI1is1xUTwMIC0W2HX00ar%2Bve2DyhpkQRqlqYgZHr9Xc26mhORJcIrULYwdUn2MleajeqVqi8QvEcIuCjYgKJL0FQdc%2B%2F2zNmmm18CwbtQfbgwguoXWDbeyqlBld7Sm9tuXDzWoCAL2nTZNo09IKRe3AS3VYZEZRE8ovoGc02mRJdtrKIcGbfLLJYABhk%2BE1mJvYMkLEVzVEJ%2B7cLRRxfM8F%2FaSB8ZKfQ4qRL%2FXoQ%2BTIlR8brKvFEp76UTkY%2FLYBfEejqcIsEuR03%2BqmJsemBNIKoDPvwCWfZOTJiT5VqdfgSiEQlaYHThK1mvvuM6epdw6X8R5gnvut7moTtusMLCbXO2ROyVUwljYRvlYmNaUtlXV33bfLCl411S9dEIO8P4v8g%2F3lAVQhh0s7z9JHSrWmJC5lA%3D%3D&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20240904T020741Z&X-Amz-SignedHeaders=host&X-Amz-Expires=28800&X-Amz-Credential=ASIA3JA3D2F343GFWJRH%2F20240904%2Fap-southeast-1%2Fs3%2Faws4_request&X-Amz-Signature=375d6d4ae9e99f7ba154665a1b98f841b737f5220fdec6a884af655d00f9b5d7" && \
# #     dpkg -i cudnn-local-repo-ubuntu2204-8.9.7.29_1.0-1_amd64.deb && \
# #     cp /var/cudnn-local-repo-*/cudnn-local-*-keyring.gpg /usr/share/keyrings/ && \
# #     apt update
# RUN --mount=type=cache,target=/root/.cache/apt apt install -y zlib1g && \
#     wget -O "cudnn-local-repo-ubuntu2204-8.8.1.3_1.0-1_amd64.deb" "https://tapway-nvidia.s3.ap-southeast-1.amazonaws.com/cudnn-local-repo-ubuntu2204-8.8.1.3_1.0-1_amd64.deb?response-content-disposition=inline&X-Amz-Security-Token=IQoJb3JpZ2luX2VjELL%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaDmFwLXNvdXRoZWFzdC0xIkYwRAIgQ%2BeOID5oX3%2FGfeTfnCcZG4prmOESkOxhDamdZAkdp%2FQCIGaBqYhunWzQ1uc39IEs79kH69elHQiW3LS4%2Bf2NsSiYKu0CCMv%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEQAhoMNzc1Mjk4NDcwMjYzIgzx%2FV9OdivFcousCTEqwQLQJ%2FY5FLHWToF2be0oZ0TaqlPPLxSYuIb0e9CAMHJpemvPB%2FgCF0Dqj3ojnW7rjXRPrlKHeQGCau%2FcApdW8IgFB3qZN%2F3QmsB0ZMenVNE3sSlEKpzdeQKyXfqOsb1SuI9rB1B6OSnmyu26JhLcURe%2FzTxXHUetOXYF1dF%2F1oyw4c2l%2FDqfmB3g8qW4s2r3GKRxI%2F7iv7EeSQm8UOSZswqftfN8e02Vt0wrCaqDVNlol2Hjvl6A0N%2FH6ArrjWrpAzc%2BjvjzVvFhev9tJLadNvbooMh1lnITQO0LmbkKyxN3nxpJVBag%2BG3QrYA%2FF4%2BXG3o8Y7I3MPCzMfxso4TLPx9aZSbMxAJcibg0kX1XSRT9%2BiBk%2BLF3QeODoVNSAY84NV3pZ45nDLSOUtYVrHBMlyu2XhuMssltVyZZiaZWBSqo2wAwkP3etgY6tAI1is1xUTwMIC0W2HX00ar%2Bve2DyhpkQRqlqYgZHr9Xc26mhORJcIrULYwdUn2MleajeqVqi8QvEcIuCjYgKJL0FQdc%2B%2F2zNmmm18CwbtQfbgwguoXWDbeyqlBld7Sm9tuXDzWoCAL2nTZNo09IKRe3AS3VYZEZRE8ovoGc02mRJdtrKIcGbfLLJYABhk%2BE1mJvYMkLEVzVEJ%2B7cLRRxfM8F%2FaSB8ZKfQ4qRL%2FXoQ%2BTIlR8brKvFEp76UTkY%2FLYBfEejqcIsEuR03%2BqmJsemBNIKoDPvwCWfZOTJiT5VqdfgSiEQlaYHThK1mvvuM6epdw6X8R5gnvut7moTtusMLCbXO2ROyVUwljYRvlYmNaUtlXV33bfLCl411S9dEIO8P4v8g%2F3lAVQhh0s7z9JHSrWmJC5lA%3D%3D&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20240904T024317Z&X-Amz-SignedHeaders=host&X-Amz-Expires=28800&X-Amz-Credential=ASIA3JA3D2F343GFWJRH%2F20240904%2Fap-southeast-1%2Fs3%2Faws4_request&X-Amz-Signature=87b1915e41f8cad7bf6a2b98a4e4bcbbfee5dfd63624de065e612a07dfb14123" && \
#     dpkg -i cudnn-local-repo-ubuntu2204-8.8.1.3_1.0-1_amd64.deb && \
#     cp /var/cudnn-local-repo-*/cudnn-local-*-keyring.gpg /usr/share/keyrings/ && \
#     apt update && \
#     apt install -y libcudnn8=8.8.1.3-1+cuda12.0 libcudnn8-dev=8.8.1.3-1+cuda12.0 && \
#     rm -rf cudnn-local-repo-ubuntu2204-8.8.1.3_1.0-1_amd64.deb /var/cudnn-local-repo-ubuntu2204-8.8.1.3 && \
#     apt purge -y --auto-remove && \
#     apt clean && \
#     rm -rf /var/lib/apt/lists/* /root/.cache/apt/*

# # # Step 7: Install Python packages and TensorFlow with CUDA support
# # RUN --mount=type=cache,target=/root/.cache/pip python${PYTHON_VERSION} -m pip install tensorflow[and-cuda] && \
# #     wget https://plus-edge-server-libs.s3.ap-southeast-1.amazonaws.com/hailo_dataflow_compiler-3.28.0-py3-none-linux_x86_64.whl && \
# #     python${PYTHON_VERSION} -m pip install hailo_dataflow_compiler-3.28.0-py3-none-linux_x86_64.whl && \
# #     rm hailo_dataflow_compiler-3.28.0-py3-none-linux_x86_64.whl && \
# #     python${PYTHON_VERSION} -m pip install nvidia-dali-cuda120 nvidia-dali-tf-plugin-cuda120 && \
# #     python${PYTHON_VERSION} -m pip cache purge && \
# #     rm -rf /root/.cache/pip/* /tmp/*

# # # Step 8: Set environment variables for CUDA 12.0
# # ENV PATH=/usr/local/cuda-12.0/bin:$PATH \
# #     LD_LIBRARY_PATH=/usr/local/cuda-12.0/lib64:$LD_LIBRARY_PATH

# # Step 9: Default command
# CMD ["/bin/bash"]

FROM --platform=linux/amd64 ubuntu:22.04

ARG PYTHON_VERSION=3.8

ENV DEBIAN_FRONTEND=noninteractive

RUN --mount=type=cache,target=/root/.cache/apt apt update && \
    apt upgrade -y && \
    apt install -y software-properties-common tzdata && \
    ln -fs /usr/share/zoneinfo/Etc/UTC /etc/localtime && \
    dpkg-reconfigure --frontend noninteractive tzdata && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt update && \
    apt install -y \
        python${PYTHON_VERSION} \
        python${PYTHON_VERSION}-distutils \
        python${PYTHON_VERSION}-dev \
        wget \
        gnupg \
        graphviz \
        libgraphviz-dev && \
    apt-mark hold nvidia-dkms-525 && apt-mark hold nvidia-driver-525 && apt-mark hold nvidia-utils-525

# python Pip
RUN --mount=type=cache,target=/root/.cache/pip wget https://bootstrap.pypa.io/get-pip.py  && \
    python${PYTHON_VERSION} get-pip.py && \
    rm get-pip.py

# Install cuda
RUN --mount=type=cache,target=/root/.cache/apt \
    apt-key del 7fa2af80 && \
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb && \
    dpkg -i cuda-keyring_1.1-1_all.deb && \
    apt update && \
    apt install -y cuda-11-8 && \
    rm -rf cuda-keyring_1.1-1_all.deb

# Install cudnn (Download from plus-edge-server-libs s3 bucket)
RUN --mount=type=cache,target=/root/.cache/apt apt install -y zlib1g && \
    wget https://plus-edge-server-libs.s3.ap-southeast-1.amazonaws.com/cudnn-local-repo-ubuntu2204-8.9.6.50_1.0-1_amd64.deb && \
    dpkg -i cudnn-local-repo-ubuntu2204-8.9.6.50_1.0-1_amd64.deb && \
    cp /var/cudnn-local-repo-ubuntu2204-8.9.6.50/cudnn-local-692B6C75-keyring.gpg /usr/share/keyrings/ && \
    apt update && \
    apt install -y libcudnn8=8.9.6.50-1+cuda11.8 libcudnn8-dev=8.9.6.50-1+cuda11.8 && \
    rm -rf cudnn-local-repo-ubuntu2204-8.9.6.50_1.0-1_amd64.deb /var/cudnn-local-repo-ubuntu2204-8.9.6.50 && \
    apt purge -y --auto-remove && \
    apt clean && \
    rm -rf /var/lib/apt/lists/* /root/.cache/apt/*

RUN --mount=type=cache,target=/root/.cache/pip python${PYTHON_VERSION} -m pip install tensorflow[and-cuda] && \
    python${PYTHON_VERSION} -m pip install --extra-index-url https://developer.download.nvidia.com/compute/redist nvidia-dali-cuda110 nvidia-dali-tf-plugin-cuda110 && \
    wget https://plus-edge-server-libs.s3.ap-southeast-1.amazonaws.com/hailo_dataflow_compiler-3.28.0-py3-none-linux_x86_64.whl && \
    python${PYTHON_VERSION} -m pip install hailo_dataflow_compiler-3.28.0-py3-none-linux_x86_64.whl && \
    rm hailo_dataflow_compiler-3.28.0-py3-none-linux_x86_64.whl && \
    python${PYTHON_VERSION} -m pip cache purge && \
    rm -rf /root/.cache/pip/* /tmp/*

# Set environment variables
ENV PATH=/usr/local/cuda-11.8/bin:$PATH \
    LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH