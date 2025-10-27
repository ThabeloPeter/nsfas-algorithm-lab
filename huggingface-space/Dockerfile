# Use pre-built image with face_recognition already installed
# This avoids building dlib from source which requires a lot of RAM
FROM animcogn/face_recognition:cpu

# Set working directory
WORKDIR /app

# Set environment variable for PaddleOCR model storage
ENV PADDLEOCR_HOME=/app/.paddleocr

# Update apt sources to use archive (Buster is archived)
RUN sed -i 's/deb.debian.org/archive.debian.org/g' /etc/apt/sources.list && \
    sed -i 's|security.debian.org|archive.debian.org|g' /etc/apt/sources.list && \
    sed -i '/stretch-updates/d' /etc/apt/sources.list && \
    sed -i '/buster-updates/d' /etc/apt/sources.list

# Install system dependencies for OpenCV and PaddleOCR
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install additional dependencies we need
# Using compatible versions to avoid typing-extensions conflict
RUN pip install --no-cache-dir \
    fastapi==0.100.0 \
    uvicorn[standard]==0.23.0 \
    PyMuPDF==1.23.8 \
    python-multipart==0.0.6 \
    opencv-python-headless==4.8.1.78 \
    mtcnn==0.1.1 \
    tensorflow-cpu==2.13.0 \
    keras==2.13.1 \
    paddlepaddle \
    paddleocr==2.7.3

# Create directory for PaddleOCR models with proper permissions
RUN mkdir -p /app/.paddleocr && chmod -R 777 /app/.paddleocr

# Copy application code and modules
COPY app.py .
COPY validators.py .
COPY ocr_extraction.py .
COPY face_extraction.py .
COPY image_processing.py .

# Expose port 7860 (HuggingFace Spaces default)
EXPOSE 7860

# Run the application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]

