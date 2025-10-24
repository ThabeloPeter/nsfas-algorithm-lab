from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import cv2
import numpy as np
from PIL import Image
import io
import base64
import fitz  # PyMuPDF
from typing import Dict, Any, List, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="NSFAS Face Extraction API - OpenCV", version="2.0.0")

# CORS middleware - allow all origins for testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize OpenCV face detector (Haar Cascade)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Verify cascade loaded successfully
if face_cascade.empty():
    logger.error("Failed to load Haar Cascade classifier")
    raise RuntimeError("Failed to load face detection model")

logger.info("OpenCV Haar Cascade face detector initialized successfully")

def pdf_to_image(pdf_bytes: bytes) -> np.ndarray:
    """Convert first page of PDF to high-quality numpy array image."""
    try:
        pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
        first_page = pdf_document[0]
        
        # Render at 3x scale for HIGH QUALITY
        # Higher scale = better quality for face detection
        mat = fitz.Matrix(3.0, 3.0)
        pix = first_page.get_pixmap(matrix=mat, alpha=False)  # No alpha channel
        
        # Convert to numpy array
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        
        # Convert to BGR (OpenCV format)
        if pix.n == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        elif pix.n == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        pdf_document.close()
        logger.info(f"PDF converted to HIGH QUALITY image: {img.shape}")
        return img
    except Exception as e:
        logger.error(f"PDF conversion error: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Failed to convert PDF: {str(e)}")

def preprocess_for_detection(gray_image: np.ndarray) -> np.ndarray:
    """
    Preprocess grayscale image to handle various lighting conditions.
    Works for both too bright (overexposed) and too dark images.
    """
    # 1. Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    # Better than regular histogram equalization - handles local lighting variations
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray_image)
    
    # 2. Denoise while preserving edges
    # Reduces noise from camera/lighting without blurring faces
    denoised = cv2.fastNlMeansDenoising(enhanced, None, h=10, templateWindowSize=7, searchWindowSize=21)
    
    # 3. Normalize contrast
    # Ensures consistent brightness range regardless of lighting
    normalized = cv2.normalize(denoised, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    
    logger.info("Image preprocessed: CLAHE + denoising + normalization applied")
    return normalized

def detect_faces_opencv(image: np.ndarray, id_type: str = 'full') -> List[Tuple[int, int, int, int]]:
    """
    Detect faces using OpenCV Haar Cascade with ROBUST preprocessing and ROI optimization.
    Handles all lighting conditions: too bright, too dark, uneven lighting.
    
    Args:
        image: Input image
        id_type: 'smart' for Smart ID (focus right side), 'green' for Green ID, 'full' for entire image
    
    Returns:
        List of face rectangles as (x, y, w, h) in FULL image coordinates
    """
    height, width = image.shape[:2]
    
    # Define Region of Interest (ROI) based on ID type
    if id_type == 'smart':
        # Smart ID Card: Photo is on RIGHT side (right 50% to be safe)
        roi_x_start = int(width * 0.50)  # Start at 50% from left
        roi_x_end = width
        roi_y_start = 0
        roi_y_end = height
        logger.info(f"Smart ID detected - Focusing on RIGHT side: x={roi_x_start}-{roi_x_end}")
    elif id_type == 'green':
        # Green ID Book: Photo typically in upper-center area
        roi_x_start = int(width * 0.15)  # 15% from left
        roi_x_end = int(width * 0.85)    # 85% from left
        roi_y_start = 0
        roi_y_end = int(height * 0.75)   # Top 75%
        logger.info(f"Green ID detected - Focusing on upper-center area")
    else:
        # Full image scan
        roi_x_start = 0
        roi_x_end = width
        roi_y_start = 0
        roi_y_end = height
        logger.info(f"Scanning full image")
    
    # Extract ROI
    roi = image[roi_y_start:roi_y_end, roi_x_start:roi_x_end]
    
    # Convert ROI to grayscale for detection
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
    # ROBUST PREPROCESSING - Handles ALL lighting conditions
    preprocessed_roi = preprocess_for_detection(gray_roi)
    
    # Detect faces in preprocessed ROI with BALANCED parameters
    faces_in_roi = face_cascade.detectMultiScale(
        preprocessed_roi,
        scaleFactor=1.05,     # Smaller steps = more thorough detection
        minNeighbors=5,       # Slightly more lenient (was 6)
        minSize=(50, 50),     # Even smaller minimum (was 60x60)
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    
    logger.info(f"OpenCV detected {len(faces_in_roi)} face(s) in preprocessed ROI")
    
    # Convert ROI coordinates back to full image coordinates
    faces = []
    if len(faces_in_roi) > 0:
        for (x, y, w, h) in faces_in_roi:
            # Adjust coordinates relative to full image
            full_x = x + roi_x_start
            full_y = y + roi_y_start
            faces.append((full_x, full_y, w, h))
    
    # Additional validation: Filter out faces that are too small relative to image
    if len(faces) > 0:
        image_area = width * height
        
        valid_faces = []
        for (x, y, w, h) in faces:
            face_area = w * h
            # Face must be at least 0.5% of image area (was 1%, too strict)
            min_area = image_area * 0.005
            if face_area >= min_area:
                valid_faces.append((x, y, w, h))
            else:
                logger.info(f"Rejected face: too small ({face_area} < {min_area})")
        
        logger.info(f"Valid faces after size filtering: {len(valid_faces)}")
        return np.array(valid_faces) if valid_faces else np.array([])
    
    return np.array(faces)

def score_faces(faces: np.ndarray, image_shape: tuple) -> list:
    """
    Score detected faces based on size and position with QUALITY CHECKS.
    Prioritizes larger faces closer to the top-center (typical ID photo position).
    
    Args:
        faces: Array of (x, y, w, h) face rectangles
        image_shape: (height, width) of image
    """
    height, width = image_shape[:2]
    scored_faces = []
    
    for idx, (x, y, w, h) in enumerate(faces):
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
        
        # QUALITY CHECK: Face aspect ratio (should be roughly square or portrait)
        aspect_ratio = w / h if h > 0 else 0
        # Ideal face aspect ratio is between 0.7 (portrait) and 1.3 (slightly wide)
        aspect_ratio_valid = 0.6 <= aspect_ratio <= 1.5
        
        # Combined score (70% size, 30% position)
        combined_score = size_score * 0.7 + pos_score * 0.3
        
        # Confidence: Higher score = higher confidence
        # Scale to percentage (multiply by 100)
        confidence = min(100, combined_score * 150)  # Boost for better display
        
        # REJECT if confidence too low or bad aspect ratio
        if confidence < 20 or not aspect_ratio_valid:  # Lowered to 20 with better preprocessing
            logger.info(f"Rejected face {idx}: confidence={confidence:.1f}%, aspect_ratio={aspect_ratio:.2f}")
            continue
        
        scored_faces.append({
            'index': idx,
            'location': (x, y, w, h),
            'area': int(face_area),
            'size_score': float(size_score),
            'pos_score': float(pos_score),
            'combined_score': float(combined_score),
            'confidence': float(confidence),
            'aspect_ratio': float(aspect_ratio)
        })
    
    # Sort by combined score (highest first)
    scored_faces.sort(key=lambda x: x['combined_score'], reverse=True)
    
    logger.info(f"Scored {len(scored_faces)} valid faces (after quality checks)")
    return scored_faces

def extract_face_region(image: np.ndarray, location: tuple, padding_percent: float = 0.3) -> np.ndarray:
    """
    Extract ONLY the face region from image with smart padding.
    
    Args:
        image: Input image
        location: (x, y, w, h) face rectangle
        padding_percent: Percentage of face size to add as padding (0.3 = 30%)
    """
    height, width = image.shape[:2]
    x, y, w, h = location
    
    # Calculate smart padding based on face size
    padding_x = int(w * padding_percent)
    padding_y = int(h * padding_percent)
    
    # Add padding but keep within image bounds
    x1 = max(0, x - padding_x)
    y1 = max(0, y - padding_y)
    x2 = min(width, x + w + padding_x)
    y2 = min(height, y + h + padding_y)
    
    # Extract ONLY the face region
    face_image = image[y1:y2, x1:x2]
    
    # Apply slight sharpening for better quality
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    face_image = cv2.filter2D(face_image, -1, kernel)
    
    logger.info(f"Extracted CROPPED face: {face_image.shape}")
    return face_image

def image_to_base64(image: np.ndarray) -> str:
    """Convert numpy image to base64 string."""
    # OpenCV uses BGR, convert to RGB for display
    if len(image.shape) == 3 and image.shape[2] == 3:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        image_rgb = image
    
    # Convert to PIL Image
    pil_image = Image.fromarray(image_rgb)
    
    # Convert to bytes
    buffer = io.BytesIO()
    pil_image.save(buffer, format='JPEG', quality=95)
    img_bytes = buffer.getvalue()
    
    # Encode to base64
    img_base64 = base64.b64encode(img_bytes).decode('utf-8')
    return f"data:image/jpeg;base64,{img_base64}"

@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "NSFAS Face Extraction API - OpenCV",
        "version": "2.0.0",
        "detector": "OpenCV Haar Cascade"
    }

@app.post("/extract-face")
async def extract_face(
    file: UploadFile = File(...),
    id_type: str = Form('full')
) -> JSONResponse:
    """
    Extract ONLY the cropped face from uploaded ID document.
    
    Process Flow:
    1. Upload PDF/Image → Server processes everything
    2. PDF converted to high-quality image (if needed)
    3. Face detection using OpenCV with ROI optimization (server-side)
    4. Extract and crop ONLY the face region
    5. Return cropped face to UI
    
    ROI Optimization:
    - Smart ID: Scans RIGHT 45% (where photo is located)
    - Green ID: Scans upper-center 60% (typical photo location)
    - Full: Scans entire image
    
    All processing on SERVER - zero stress on client device!
    
    Args:
        file: ID document (image or PDF)
        id_type: 'smart', 'green', or 'full' for ROI optimization
    
    Returns:
        - face_image: Base64 encoded CROPPED face only
        - metadata: Detection and scoring information
    """
    try:
        # Read file content
        contents = await file.read()
        logger.info(f"Processing file: {file.filename}, size: {len(contents)} bytes")
        
        # Check file size (max 10MB)
        if len(contents) > 10 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="File too large. Maximum 10MB allowed.")
        
        # Determine file type and load image
        if file.content_type == 'application/pdf':
            image = pdf_to_image(contents)
        elif file.content_type in ['image/jpeg', 'image/png', 'image/jpg']:
            # Load image from bytes
            nparr = np.frombuffer(contents, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if image is None:
                raise HTTPException(status_code=400, detail="Invalid image file.")
        else:
            raise HTTPException(
                status_code=400, 
                detail="Unsupported file type. Please upload JPEG, PNG, or PDF."
            )
        
        logger.info(f"Image loaded: {image.shape}")
        
        # Resize if too large (for faster processing)
        max_dimension = 1600
        height, width = image.shape[:2]
        if max(height, width) > max_dimension:
            scale = max_dimension / max(height, width)
            new_width = int(width * scale)
            new_height = int(height * scale)
            image_resized = cv2.resize(image, (new_width, new_height))
            logger.info(f"Resized image to: {image_resized.shape}")
        else:
            image_resized = image
            scale = 1.0
        
        # Detect faces using OpenCV with strict filtering and ROI optimization
        faces = detect_faces_opencv(image_resized, id_type=id_type)
        
        if len(faces) == 0:
            raise HTTPException(
                status_code=404, 
                detail="No face detected in the document. Please ensure the ID photo is clearly visible and try again."
            )
        
        # Score faces and select best one (with quality checks)
        scored_faces = score_faces(faces, image_resized.shape)
        
        if len(scored_faces) == 0:
            raise HTTPException(
                status_code=404, 
                detail="No valid face detected. The detected regions do not appear to be actual faces. Please ensure the ID photo is clearly visible."
            )
        
        best_face = scored_faces[0]
        
        # Final confidence check: Reject if too low
        if best_face.get('confidence', 0) < 25:  # Lowered to 25 with robust preprocessing
            raise HTTPException(
                status_code=404,
                detail=f"Face detection confidence too low ({best_face.get('confidence', 0):.1f}%). Please try repositioning the ID or adjusting lighting."
            )
        
        # Scale back to original image coordinates
        if scale != 1.0:
            x, y, w, h = best_face['location']
            x = int(x / scale)
            y = int(y / scale)
            w = int(w / scale)
            h = int(h / scale)
            original_location = (x, y, w, h)
        else:
            original_location = best_face['location']
        
        # Extract ONLY the cropped face from original resolution image
        # All processing done on server - no stress on client device
        face_image = extract_face_region(image, original_location, padding_percent=0.3)
        
        # Convert to base64
        face_base64 = image_to_base64(face_image)
        
        # Calculate face dimensions
        face_height, face_width = face_image.shape[:2]
        
        # Prepare metadata
        metadata = {
            'total_faces': len(faces),
            'valid_faces': len(scored_faces),
            'selected_index': best_face['index'],
            'confidence': round(best_face.get('confidence', best_face['combined_score'] * 100), 1),
            'face_size': {
                'width': face_width,
                'height': face_height
            },
            'position_score': round(best_face['pos_score'] * 100, 1),
            'size_score': round(best_face['size_score'] * 100, 1),
            'combined_score': round(best_face['combined_score'] * 100, 1),
            'aspect_ratio': round(best_face.get('aspect_ratio', 1.0), 2),
            'detector': 'OpenCV Haar Cascade (Strict Mode)',
            'all_faces': [
                {
                    'index': f['index'],
                    'area': f['area'],
                    'score': round(f['combined_score'] * 100, 1),
                    'confidence': round(f.get('confidence', f['combined_score'] * 100), 1)
                }
                for f in scored_faces
            ]
        }
        
        logger.info(f"Successfully extracted face. Metadata: {metadata}")
        
        return JSONResponse(content={
            'success': True,
            'face_image': face_base64,
            'metadata': metadata
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing file: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/compare-faces")
async def compare_faces(
    selfie: UploadFile = File(...),
    id_photo: UploadFile = File(...)
) -> JSONResponse:
    """
    Compare two face images using OpenCV template matching.
    
    Returns:
        - match: Boolean indicating if faces match
        - similarity: Similarity score (0-100)
        - confidence: Match confidence percentage
    """
    try:
        # Load both images
        selfie_bytes = await selfie.read()
        id_bytes = await id_photo.read()
        
        # Decode images
        selfie_arr = np.frombuffer(selfie_bytes, np.uint8)
        id_arr = np.frombuffer(id_bytes, np.uint8)
        
        selfie_img = cv2.imdecode(selfie_arr, cv2.IMREAD_COLOR)
        id_img = cv2.imdecode(id_arr, cv2.IMREAD_COLOR)
        
        if selfie_img is None or id_img is None:
            raise HTTPException(status_code=400, detail="Invalid image files.")
        
        # Detect faces in both images
        selfie_faces = detect_faces_opencv(selfie_img)
        id_faces = detect_faces_opencv(id_img)
        
        if len(selfie_faces) == 0:
            raise HTTPException(status_code=404, detail="No face detected in selfie.")
        
        if len(id_faces) == 0:
            raise HTTPException(status_code=404, detail="No face detected in ID photo.")
        
        # Extract first face from each
        sx, sy, sw, sh = selfie_faces[0]
        ix, iy, iw, ih = id_faces[0]
        
        selfie_face = selfie_img[sy:sy+sh, sx:sx+sw]
        id_face = id_img[iy:iy+ih, ix:ix+iw]
        
        # Resize to same size for comparison
        target_size = (100, 100)
        selfie_resized = cv2.resize(selfie_face, target_size)
        id_resized = cv2.resize(id_face, target_size)
        
        # Convert to grayscale
        selfie_gray = cv2.cvtColor(selfie_resized, cv2.COLOR_BGR2GRAY)
        id_gray = cv2.cvtColor(id_resized, cv2.COLOR_BGR2GRAY)
        
        # Calculate histogram similarity
        selfie_hist = cv2.calcHist([selfie_gray], [0], None, [256], [0, 256])
        id_hist = cv2.calcHist([id_gray], [0], None, [256], [0, 256])
        
        # Normalize histograms
        cv2.normalize(selfie_hist, selfie_hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        cv2.normalize(id_hist, id_hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        
        # Compare histograms (correlation method)
        similarity = cv2.compareHist(selfie_hist, id_hist, cv2.HISTCMP_CORREL)
        
        # Convert to percentage
        similarity_percent = max(0, min(100, similarity * 100))
        
        # Match threshold (70%)
        threshold = 0.7
        is_match = similarity > threshold
        
        logger.info(f"Face comparison - Similarity: {similarity:.4f}, Match: {is_match}")
        
        return JSONResponse(content={
            'success': True,
            'match': bool(is_match),
            'similarity': round(similarity_percent, 1),
            'confidence': round(similarity_percent, 1),
            'threshold': threshold * 100,
            'method': 'OpenCV Histogram Correlation'
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error comparing faces: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
