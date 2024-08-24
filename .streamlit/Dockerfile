FROM python:3.10-slim

WORKDIR /app

# Cài đặt system dependencies
COPY packages.txt .
RUN apt-get update && cat packages.txt | xargs apt-get install -y

# Copy requirements và cài đặt Python packages
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy source code
COPY . .

# Cập nhật ldconfig
RUN ldconfig

# Kiểm tra OpenCV
RUN python -c "import cv2; print(cv2.__version__)"

# Chạy setup script
RUN chmod +x setup.sh
RUN ./setup.sh

# Chạy ứng dụng
CMD streamlit run Geospatial_Comparisons.py