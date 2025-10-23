from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import face_recognition
import cv2
import numpy as np
from PIL import Image
import io
import base64
import fitz  # PyMuPDF
from typing import Dict, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="NSFAS Face Extraction API", version="1.0.0")

# CORS middleware - allow all origins for preprod testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def pdf_to_image(pdf_bytes: bytes) -> np.ndarray:
    """Convert first page of PDF to numpy array image."""
    try:
        pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
        first_page = pdf_document[0]
        
        # Render at 2x scale for better quality
        mat = fitz.Matrix(2.0, 2.0)
        pix = first_page.get_pixmap(matrix=mat)
        
        # Convert to numpy array
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        
        # Convert RGBA to RGB if needed
        if pix.n == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        
        pdf_document.close()
        logger.info(f"PDF converted to image: {img.shape}")
        return img
    except Exception as e:
        logger.error(f"PDF conversion error: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Failed to convert PDF: {str(e)}")

def score_faces(face_locations: list, image_shape: tuple) -> list:
    """
    Score detected faces based on size and position.
    Prioritizes larger faces closer to the top-center (typical ID photo position).
    """
    height, width = image_shape[:2]
    scored_faces = []
    
    for idx, (top, right, bottom, left) in enumerate(face_locations):
        # Calculate face dimensions
        face_width = right - left
        face_height = bottom - top
        face_area = face_width * face_height
        
        # Calculate center position
        center_x = left + face_width / 2
        center_y = top + face_height / 2
        
        # Distance from ideal position (top-center to upper-third)
        ideal_x = width / 2
        ideal_y = height / 3
        dist_from_ideal = np.sqrt((center_x - ideal_x)**2 + (center_y - ideal_y)**2)
        max_dist = np.sqrt((width / 2)**2 + (height / 3)**2)
        
        # Position score (0-1, higher is better)
        pos_score = 1 - (dist_from_ideal / max_dist)
        
        # Size score (0-1, higher is better)
        size_score = face_area / (width * height)
        
        # Combined score (70% size, 30% position)
        combined_score = size_score * 0.7 + pos_score * 0.3
        
        scored_faces.append({
            'index': idx,
            'location': (top, right, bottom, left),
            'area': int(face_area),
            'size_score': float(size_score),
            'pos_score': float(pos_score),
            'combined_score': float(combined_score)
        })
    
    # Sort by combined score (highest first)
    scored_faces.sort(key=lambda x: x['combined_score'], reverse=True)
    
    logger.info(f"Scored {len(scored_faces)} faces")
    return scored_faces

def extract_face_region(image: np.ndarray, location: tuple, padding: int = 30) -> np.ndarray:
    """Extract face region from image with padding."""
    height, width = image.shape[:2]
    top, right, bottom, left = location
    
    # Add padding
    top = max(0, top - padding)
    left = max(0, left - padding)
    bottom = min(height, bottom + padding)
    right = min(width, right + padding)
    
    # Extract region
    face_image = image[top:bottom, left:right]
    logger.info(f"Extracted face region: {face_image.shape}")
    return face_image

def image_to_base64(image: np.ndarray) -> str:
    """Convert numpy image to base64 string."""
    # Convert BGR to RGB if needed
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
        "service": "NSFAS Face Extraction API",
        "version": "1.0.0"
    }

@app.post("/extract-face")
async def extract_face(file: UploadFile = File(...)) -> JSONResponse:
    """
    Extract face from uploaded ID document (image or PDF).
    
    Returns:
        - face_image: Base64 encoded extracted face
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
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
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
        
        # Detect faces
        face_locations = face_recognition.face_locations(image_resized, model='hog')
        
        if len(face_locations) == 0:
            raise HTTPException(
                status_code=404, 
                detail="No face detected in the document. Please upload a clearer ID photo."
            )
        
        logger.info(f"Detected {len(face_locations)} face(s)")
        
        # Score faces and select best one
        scored_faces = score_faces(face_locations, image_resized.shape)
        best_face = scored_faces[0]
        
        # Scale back to original image coordinates
        if scale != 1.0:
            top, right, bottom, left = best_face['location']
            top = int(top / scale)
            right = int(right / scale)
            bottom = int(bottom / scale)
            left = int(left / scale)
            original_location = (top, right, bottom, left)
        else:
            original_location = best_face['location']
        
        # Extract face from original resolution image
        face_image = extract_face_region(image, original_location, padding=30)
        
        # Convert to base64
        face_base64 = image_to_base64(face_image)
        
        # Calculate face dimensions
        face_height, face_width = face_image.shape[:2]
        
        # Prepare metadata
        metadata = {
            'total_faces': len(face_locations),
            'selected_index': best_face['index'],
            'confidence': round(best_face['combined_score'] * 100, 1),
            'face_size': {
                'width': face_width,
                'height': face_height
            },
            'position_score': round(best_face['pos_score'] * 100, 1),
            'size_score': round(best_face['size_score'] * 100, 1),
            'combined_score': round(best_face['combined_score'] * 100, 1),
            'all_faces': [
                {
                    'index': f['index'],
                    'area': f['area'],
                    'score': round(f['combined_score'] * 100, 1)
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
    Compare two face images (selfie vs ID photo).
    
    Returns:
        - match: Boolean indicating if faces match
        - distance: Face distance (lower is better, <0.6 is a match)
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
        
        # Convert BGR to RGB
        selfie_rgb = cv2.cvtColor(selfie_img, cv2.COLOR_BGR2RGB)
        id_rgb = cv2.cvtColor(id_img, cv2.COLOR_BGR2RGB)
        
        # Get face encodings
        selfie_encodings = face_recognition.face_encodings(selfie_rgb)
        id_encodings = face_recognition.face_encodings(id_rgb)
        
        if len(selfie_encodings) == 0:
            raise HTTPException(status_code=404, detail="No face detected in selfie.")
        
        if len(id_encodings) == 0:
            raise HTTPException(status_code=404, detail="No face detected in ID photo.")
        
        # Use first face from each image
        selfie_encoding = selfie_encodings[0]
        id_encoding = id_encodings[0]
        
        # Calculate face distance
        distance = face_recognition.face_distance([id_encoding], selfie_encoding)[0]
        
        # Determine match (threshold: 0.6)
        threshold = 0.6
        is_match = distance < threshold
        
        # Calculate confidence (inverse of distance, normalized)
        confidence = max(0, min(100, (1 - distance) * 100))
        
        logger.info(f"Face comparison - Distance: {distance:.4f}, Match: {is_match}")
        
        return JSONResponse(content={
            'success': True,
            'match': bool(is_match),
            'distance': float(distance),
            'confidence': round(confidence, 1),
            'threshold': threshold
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error comparing faces: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)


