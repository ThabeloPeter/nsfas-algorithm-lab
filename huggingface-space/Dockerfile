# Use pre-built image with face_recognition already installed
# This avoids building dlib from source which requires a lot of RAM
FROM animcogn/face_recognition:cpu

# Set working directory
WORKDIR /app

# Install additional dependencies we need
RUN pip install --no-cache-dir \
    fastapi==0.104.1 \
    uvicorn[standard]==0.24.0 \
    PyMuPDF==1.23.8 \
    python-multipart==0.0.6 \
    opencv-python-headless==4.8.1.78 \
    mtcnn==0.1.1 \
    tensorflow-cpu==2.13.0 \
    keras==2.13.1

# Copy application code
COPY app.py .

# Expose port 7860 (HuggingFace Spaces default)
EXPOSE 7860

# Run the application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]

