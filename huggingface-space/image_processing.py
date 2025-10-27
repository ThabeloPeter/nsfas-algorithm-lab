"""
Image processing utilities for ID document handling.
Handles image conversion, format detection, and preprocessing.
"""

import cv2
import numpy as np
import fitz  # PyMuPDF
from PIL import Image
import io
import base64
from typing import Tuple
from fastapi import HTTPException
import logging

logger = logging.getLogger(__name__)


def pdf_to_image(pdf_bytes: bytes) -> np.ndarray:
    """
    Convert first page of PDF to high-quality numpy array image.
    
    Args:
        pdf_bytes: PDF file as bytes
    
    Returns:
        Image as numpy array in BGR format
    
    Raises:
        HTTPException: If PDF conversion fails
    """
    try:
        pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
        first_page = pdf_document[0]
        
        # Render at 3x scale for HIGH QUALITY
        mat = fitz.Matrix(3.0, 3.0)
        pix = first_page.get_pixmap(matrix=mat, alpha=False)
        
        # Convert to numpy array
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        
        # Convert to BGR (OpenCV format)
        if pix.n == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        elif pix.n == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        pdf_document.close()
        logger.info(f"PDF converted to image: {img.shape}")
        return img
    except Exception as e:
        logger.error(f"PDF conversion error: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Failed to convert PDF: {str(e)}")


def image_to_base64(image: np.ndarray) -> str:
    """
    Convert numpy image to base64 data URL string.
    
    Args:
        image: Image as numpy array (BGR or grayscale)
    
    Returns:
        Base64 data URL string
    """
    if len(image.shape) == 3 and image.shape[2] == 3:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        image_rgb = image
    
    pil_image = Image.fromarray(image_rgb)
    buffer = io.BytesIO()
    pil_image.save(buffer, format='JPEG', quality=95)
    img_bytes = buffer.getvalue()
    img_base64 = base64.b64encode(img_bytes).decode('utf-8')
    
    return f"data:image/jpeg;base64,{img_base64}"


def base64_to_blob(base64_string: str) -> bytes:
    """
    Convert base64 string to binary blob.
    
    Args:
        base64_string: Base64 encoded string (with or without data URL prefix)
    
    Returns:
        Binary image data
    """
    if ',' in base64_string:
        # Remove data URL prefix if present
        base64_string = base64_string.split(',')[1]
    
    return base64.b64decode(base64_string)


def resize_if_needed(image: np.ndarray, max_dimension: int = 1600) -> Tuple[np.ndarray, float]:
    """
    Resize image if it exceeds maximum dimension while maintaining aspect ratio.
    
    Args:
        image: Input image
        max_dimension: Maximum width or height
    
    Returns:
        Tuple of (resized_image, scale_factor)
    """
    height, width = image.shape[:2]
    
    if max(height, width) > max_dimension:
        scale = max_dimension / max(height, width)
        new_width = int(width * scale)
        new_height = int(height * scale)
        resized_image = cv2.resize(image, (new_width, new_height))
        logger.info(f"📏 Resized from {width}x{height} to {new_width}x{new_height}")
        return resized_image, scale
    
    return image, 1.0


def validate_image_file(file_bytes: bytes, content_type: str, max_size_mb: int = 10) -> np.ndarray:
    """
    Validate and load image file.
    
    Args:
        file_bytes: File content as bytes
        content_type: MIME type of file
        max_size_mb: Maximum file size in MB
    
    Returns:
        Image as numpy array in BGR format
    
    Raises:
        HTTPException: If file is invalid or too large
    """
    # Check file size
    if len(file_bytes) > max_size_mb * 1024 * 1024:
        raise HTTPException(status_code=400, detail=f"File too large. Maximum {max_size_mb}MB.")
    
    # Load image based on content type
    if content_type == 'application/pdf':
        image = pdf_to_image(file_bytes)
    elif content_type in ['image/jpeg', 'image/png', 'image/jpg']:
        nparr = np.frombuffer(file_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file.")
        logger.info(f"📐 Image loaded: {image.shape}")
    else:
        raise HTTPException(status_code=400, detail="Unsupported file type. Upload JPEG, PNG, or PDF.")
    
    return image

