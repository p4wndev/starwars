#!/bin/bash

# Cập nhật system
sudo apt-get update
sudo apt-get upgrade -y

# Cài đặt dependencies
sudo apt-get install -y libgl1-mesa-glx libglib2.0-0 libsm6 libxrender1 libxext6 libxext-dev libx11-dev mesa-utils

# Cài đặt Python packages
pip3 install --upgrade pip
pip3 install --user -r requirements.txt
pip3 install --user streamlit

# Gỡ cài đặt opencv-python hiện tại
pip3 uninstall -y opencv-python

# Cài đặt opencv-contrib-python
pip3 install --user --no-cache-dir opencv-contrib-python

# Cập nhật ldconfig
sudo ldconfig

# Kiểm tra cài đặt
python3 -c "import cv2; print(cv2.__version__)"

# Kiểm tra OpenGL
glxinfo | grep "OpenGL version"