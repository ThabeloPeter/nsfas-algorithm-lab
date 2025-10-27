"""
NSFAS Face Extraction API - Main Application
Modular architecture with separated concerns.
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging

# Import our custom modules
from face_extraction import initialize_face_detector, assess_image_quality, detect_faces, score_faces, extract_face_region
from ocr_extraction import initialize_ocr, extract_id_data
from image_processing import validate_image_file, resize_if_needed, image_to_base64
import face_recognition

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="NSFAS Face Extraction & OCR API", version="4.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize detectors
if not initialize_face_detector():
    raise RuntimeError("Failed to load MTCNN face detector")

if not initialize_ocr():
    raise RuntimeError("Failed to load PaddleOCR")


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "NSFAS Face Extraction & OCR API",
        "version": "4.0.0",
        "features": {
            "face_detection": "MTCNN Deep Learning (95-98% accuracy)",
            "face_comparison": "dlib ResNet (99.38% LFW accuracy)",
            "ocr": "PaddleOCR with SA ID validation"
        }
    }


@app.post("/extract-face")
async def extract_face(
    file: UploadFile = File(...),
    id_type: str = Form('full')
) -> JSONResponse:
    """
    Extract face from ID document and perform OCR.
    
    Features:
    - Face detection with MTCNN (95-98% accuracy)
    - Facial landmark detection
    - Eye validation (eliminates false positives)
    - Image quality assessment
    - OCR extraction of ID fields
    - SA ID number validation with checksum
    
    Args:
        file: ID document image (JPEG, PNG) or PDF
        id_type: 'smart', 'green', or 'full' for ROI optimization
    
    Returns:
        JSON response with face image, metadata, and OCR data
    """
    try:
        # Read and validate file
        contents = await file.read()
        logger.info(f"📄 Processing: {file.filename} ({len(contents)} bytes), Type: {id_type}")
        
        image = validate_image_file(contents, file.content_type)
        
        # Assess image quality FIRST
        quality = assess_image_quality(image)
        logger.info(f"🔍 Image quality: {quality['quality']} - {quality.get('issues', [])}")
        
        # Resize if too large
        image_resized, scale = resize_if_needed(image, max_dimension=1600)
        
        # Detect faces with MTCNN
        faces = detect_faces(image_resized, id_type=id_type)
        
        if len(faces) == 0:
            # Provide specific feedback based on image quality
            if quality['is_blurry']:
                detail = "Image is too blurry. Hold the camera steady and try again."
            elif quality['is_too_dark']:
                detail = "Image is too dark. Turn on the flash or move to a brighter area."
            elif quality['is_too_bright']:
                detail = "Image is overexposed. Move away from direct light sources."
            else:
                detail = "No face detected. Ensure the ID photo is clearly visible and try again."
            
            raise HTTPException(status_code=404, detail=detail)
        
        # Score and select best face
        scored_faces = score_faces(faces, image_resized.shape)
        best_face = scored_faces[0]
        
        # Scale back to original coordinates
        if scale != 1.0:
            x, y, w, h = best_face['box']
            original_box = (int(x/scale), int(y/scale), int(w/scale), int(h/scale))
        else:
            original_box = best_face['box']
        
        # Extract face region
        face_image = extract_face_region(image, original_box, padding_percent=0.3)
        face_base64 = image_to_base64(face_image)
        
        # Run OCR extraction on the original image
        logger.info("🔍 Starting OCR extraction...")
        ocr_result = extract_id_data(image, id_type)
        
        # Prepare metadata
        metadata = {
            'total_faces': len(faces),
            'selected_index': 0,
            'confidence': round(best_face['confidence'] * 100, 1),
            'face_size': {'width': face_image.shape[1], 'height': face_image.shape[0]},
            'position_score': round(best_face['pos_score'] * 100, 1),
            'size_score': round(best_face['size_score'] * 100, 1),
            'combined_score': round(best_face['combined_score'] * 100, 1),
            'aspect_ratio': round(best_face['aspect_ratio'], 2),
            'detector': 'MTCNN Deep Learning',
            'landmarks_detected': True,
            'image_quality': quality,
            'all_faces': [
                {
                    'index': idx,
                    'confidence': round(f['confidence'] * 100, 1),
                    'score': round(f['combined_score'] * 100, 1)
                }
                for idx, f in enumerate(scored_faces)
            ]
        }
        
        logger.info(f"✅ Success! Face: {metadata['confidence']}%, OCR: {ocr_result.get('fieldsExtracted', 0)}/{ocr_result.get('totalFields', 0)}")
        
        return JSONResponse(content={
            'success': True,
            'face_image': face_base64,
            'metadata': metadata,
            'ocr_data': ocr_result
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.post("/compare-faces")
async def compare_faces(
    selfie: UploadFile = File(...),
    id_photo: UploadFile = File(...)
) -> JSONResponse:
    """
    Compare two face images using face_recognition (dlib ResNet).
    Accuracy: 99.38% on LFW benchmark.
    
    Args:
        selfie: Selfie photo
        id_photo: ID photo (extracted face)
    
    Returns:
        JSON response with match result and confidence
    """
    try:
        # Load both images
        selfie_bytes = await selfie.read()
        id_bytes = await id_photo.read()
        
        # Convert to RGB format for face_recognition
        import io
        selfie_img = face_recognition.load_image_file(io.BytesIO(selfie_bytes))
        id_img = face_recognition.load_image_file(io.BytesIO(id_bytes))
        
        logger.info("🔍 Detecting faces in selfie and ID photo...")
        
        # Get face encodings (128-dimensional embeddings)
        selfie_encodings = face_recognition.face_encodings(selfie_img)
        id_encodings = face_recognition.face_encodings(id_img)
        
        if len(selfie_encodings) == 0:
            raise HTTPException(
                status_code=404, 
                detail="No face detected in selfie. Please ensure your face is clearly visible."
            )
        
        if len(id_encodings) == 0:
            raise HTTPException(
                status_code=404, 
                detail="No face detected in ID photo. Please use the extracted face from your ID."
            )
        
        # Compare faces using Euclidean distance
        distances = face_recognition.face_distance([id_encodings[0]], selfie_encodings[0])
        distance = float(distances[0])
        
        # Convert distance to similarity percentage
        similarity = max(0, min(100, (1 - distance) * 100))
        
        # Determine match based on threshold
        threshold = 0.6  # Industry standard
        is_match = distance < threshold
        
        # Calculate confidence level
        if distance < 0.4:
            confidence_level = "Very High"
            confidence = 95
        elif distance < 0.5:
            confidence_level = "High"
            confidence = 85
        elif distance < 0.6:
            confidence_level = "Medium"
            confidence = 75
        else:
            confidence_level = "Low"
            confidence = max(0, 70 - (distance - 0.6) * 100)
        
        logger.info(f"✅ Match: {is_match}, Distance: {distance:.4f}, Similarity: {similarity:.1f}%")
        
        return JSONResponse(content={
            'success': True,
            'match': bool(is_match),
            'similarity': round(similarity, 1),
            'confidence': round(confidence, 1),
            'confidence_level': confidence_level,
            'distance': round(distance, 4),
            'threshold': threshold,
            'method': 'face_recognition (dlib ResNet)',
            'accuracy': '99.38% (LFW benchmark)'
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error comparing faces: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Face comparison error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)

