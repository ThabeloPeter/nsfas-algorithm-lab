"""
Face extraction module for ID documents.
Handles face detection, validation, and extraction using MTCNN.
"""

import cv2
import numpy as np
from typing import Dict, Any, List
from mtcnn import MTCNN
import logging

logger = logging.getLogger(__name__)

# Initialize MTCNN and OpenCV cascades (will be set by init function)
detector = None
face_cascade = None
eye_cascade = None


def initialize_face_detector():
    """Initialize MTCNN face detector and OpenCV cascades."""
    global detector, face_cascade, eye_cascade
    
    try:
        detector = MTCNN()
        logger.info("✅ MTCNN face detector initialized successfully")
        
        # Keep OpenCV as fallback for eye validation
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        
        return True
    except Exception as e:
        logger.error(f"❌ Failed to initialize MTCNN: {str(e)}")
        return False


def assess_image_quality(image: np.ndarray) -> Dict[str, Any]:
    """
    Assess image quality for face detection.
    
    Args:
        image: Input image in BGR format
    
    Returns:
        Dictionary with quality metrics and recommendations
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 1. Blur detection (Laplacian variance)
    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
    is_blurry = blur_score < 100
    
    # 2. Lighting check
    mean_brightness = np.mean(gray)
    is_too_dark = mean_brightness < 50
    is_too_bright = mean_brightness > 200
    
    # 3. Contrast check
    contrast = gray.std()
    is_low_contrast = contrast < 30
    
    quality_status = 'good'
    issues = []
    
    if is_blurry:
        quality_status = 'poor'
        issues.append('Image is blurry - hold camera steady')
    if is_too_dark:
        quality_status = 'poor'
        issues.append('Too dark - turn on flash or add light')
    if is_too_bright:
        quality_status = 'poor'
        issues.append('Too bright - avoid direct light')
    if is_low_contrast:
        quality_status = 'fair'
        issues.append('Low contrast - adjust lighting')
    
    return {
        'blur_score': float(blur_score),
        'is_blurry': bool(is_blurry),
        'brightness': float(mean_brightness),
        'is_too_dark': bool(is_too_dark),
        'is_too_bright': bool(is_too_bright),
        'contrast': float(contrast),
        'is_low_contrast': bool(is_low_contrast),
        'quality': quality_status,
        'issues': issues
    }


def validate_face_with_eyes(face_region: np.ndarray) -> bool:
    """
    Validate that detected region is actually a face by detecting eyes.
    Eliminates false positives (dots, logos, etc.)
    
    Args:
        face_region: Cropped face region in BGR format
    
    Returns:
        True if face has eyes, False otherwise
    """
    gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
    eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(20, 20))
    
    # Real faces should have at least 1 eye (profile) or 2 (frontal)
    has_eyes = len(eyes) >= 1
    logger.info(f"Eye validation: {len(eyes)} eyes detected - {'VALID' if has_eyes else 'INVALID'}")
    
    return has_eyes


def detect_faces(image: np.ndarray, id_type: str = 'full') -> List[Dict[str, Any]]:
    """
    Detect faces using MTCNN deep learning detector.
    
    Args:
        image: Input image (BGR format)
        id_type: 'smart', 'green', or 'full' for ROI optimization
    
    Returns:
        List of face dictionaries with location, confidence, landmarks
    """
    height, width = image.shape[:2]
    
    # Define Region of Interest (ROI) based on ID type
    if id_type == 'smart':
        # Smart ID Card: Photo is on RIGHT side
        roi_x_start = int(width * 0.50)
        roi_x_end = width
        roi_y_start = 0
        roi_y_end = height
        logger.info(f"Smart ID - Scanning RIGHT 50%")
    elif id_type == 'green':
        # Green ID Book: Photo in upper-center
        roi_x_start = int(width * 0.15)
        roi_x_end = int(width * 0.85)
        roi_y_start = 0
        roi_y_end = int(height * 0.75)
        logger.info(f"Green ID - Scanning upper-center")
    else:
        # Full image
        roi_x_start = 0
        roi_x_end = width
        roi_y_start = 0
        roi_y_end = height
        logger.info(f"Scanning full image")
    
    # Extract ROI
    roi = image[roi_y_start:roi_y_end, roi_x_start:roi_x_end]
    
    # Convert BGR to RGB (MTCNN expects RGB)
    rgb_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    
    # Detect faces with MTCNN
    detections = detector.detect_faces(rgb_roi)
    
    logger.info(f"🧠 MTCNN detected {len(detections)} face(s)")
    
    # Filter and validate faces
    validated_faces = []
    
    for detection in detections:
        confidence = detection['confidence']
        box = detection['box']  # [x, y, width, height]
        keypoints = detection['keypoints']  # eyes, nose, mouth
        
        # Only accept high-confidence detections
        if confidence < 0.95:
            logger.info(f"Rejected: Low confidence ({confidence:.2f})")
            continue
        
        # Extract box coordinates
        x, y, w, h = box
        
        # Handle negative coordinates (MTCNN bug)
        x = max(0, x)
        y = max(0, y)
        w = min(w, roi.shape[1] - x)
        h = min(h, roi.shape[0] - y)
        
        # Convert ROI coordinates to full image coordinates
        full_x = x + roi_x_start
        full_y = y + roi_y_start
        
        # Validate aspect ratio (faces should be roughly square/portrait)
        aspect_ratio = w / h if h > 0 else 0
        if aspect_ratio < 0.5 or aspect_ratio > 1.8:
            logger.info(f"Rejected: Bad aspect ratio ({aspect_ratio:.2f})")
            continue
        
        # Extract face region for eye validation
        face_region = image[full_y:full_y+h, full_x:full_x+w]
        
        # Validate with eye detection (double-check)
        if not validate_face_with_eyes(face_region):
            logger.info(f"Rejected: No eyes detected")
            continue
        
        validated_faces.append({
            'box': (full_x, full_y, w, h),
            'confidence': confidence,
            'keypoints': {
                'left_eye': (keypoints['left_eye'][0] + roi_x_start, keypoints['left_eye'][1] + roi_y_start),
                'right_eye': (keypoints['right_eye'][0] + roi_x_start, keypoints['right_eye'][1] + roi_y_start),
                'nose': (keypoints['nose'][0] + roi_x_start, keypoints['nose'][1] + roi_y_start),
                'mouth_left': (keypoints['mouth_left'][0] + roi_x_start, keypoints['mouth_left'][1] + roi_y_start),
                'mouth_right': (keypoints['mouth_right'][0] + roi_x_start, keypoints['mouth_right'][1] + roi_y_start),
            },
            'aspect_ratio': aspect_ratio
        })
    
    logger.info(f"✅ {len(validated_faces)} validated faces (after quality checks)")
    return validated_faces


def score_faces(faces: List[Dict], image_shape: tuple) -> List[Dict]:
    """
    Score and rank faces based on size, position, and confidence.
    
    Args:
        faces: List of face dictionaries
        image_shape: Original image shape (height, width, channels)
    
    Returns:
        Sorted list of faces with scores
    """
    height, width = image_shape[:2]
    
    for face in faces:
        x, y, w, h = face['box']
        
        # Calculate face area
        face_area = w * h
        
        # Calculate center position
        center_x = x + w / 2
        center_y = y + h / 2
        
        # Distance from ideal position (top-center to upper-third)
        ideal_x = width / 2
        ideal_y = height / 3
        dist_from_ideal = np.sqrt((center_x - ideal_x)**2 + (center_y - ideal_y)**2)
        max_dist = np.sqrt((width / 2)**2 + (height / 3)**2)
        
        # Position score (0-1, higher is better)
        pos_score = 1 - (dist_from_ideal / max_dist)
        
        # Size score (0-1, higher is better)
        size_score = face_area / (width * height)
        
        # Combined score (50% confidence, 30% size, 20% position)
        combined_score = (face['confidence'] * 0.5) + (size_score * 0.3) + (pos_score * 0.2)
        
        face['size_score'] = float(size_score)
        face['pos_score'] = float(pos_score)
        face['combined_score'] = float(combined_score)
        face['area'] = int(face_area)
    
    # Sort by combined score (highest first)
    faces.sort(key=lambda x: x['combined_score'], reverse=True)
    
    return faces


def extract_face_region(image: np.ndarray, location: tuple, padding_percent: float = 0.3) -> np.ndarray:
    """
    Extract face region with padding and sharpening.
    
    Args:
        image: Input image in BGR format
        location: Face location as (x, y, width, height)
        padding_percent: Percentage of padding to add around face
    
    Returns:
        Extracted face image
    """
    height, width = image.shape[:2]
    x, y, w, h = location
    
    # Add padding
    padding_x = int(w * padding_percent)
    padding_y = int(h * padding_percent)
    
    x1 = max(0, x - padding_x)
    y1 = max(0, y - padding_y)
    x2 = min(width, x + w + padding_x)
    y2 = min(height, y + h + padding_y)
    
    face_image = image[y1:y2, x1:x2]
    
    # Apply slight sharpening
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    face_image = cv2.filter2D(face_image, -1, kernel)
    
    logger.info(f"Extracted face: {face_image.shape}")
    return face_image

