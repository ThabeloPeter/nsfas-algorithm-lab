# Face Comparison Testing (Preprod)

This is a testing component for face comparison between selfies and ID document photos. It's designed to test the face recognition and comparison functionality without interfering with the main application.

## Features

- **Selfie Capture**: Real-time face detection using webcam with face-api.js
- **ID Document Upload**: Support for JPEG, PNG, and PDF files
- **Photo Extraction**: Automatic extraction of face photo from ID document
- **Face Comparison**: Advanced face recognition comparison using face descriptors
- **Results Display**: Visual comparison results with similarity scores

## How to Access

Navigate to `/preprod` in your browser to access the testing component.

## How It Works

### 1. Selfie Capture
- Uses webcam to capture a selfie
- Real-time face detection ensures a clear face is captured
- Face detection overlay guides user positioning

### 2. ID Document Upload
- Upload ID document (JPEG, PNG, PDF)
- **Automatic PDF to Image Conversion**: PDFs are automatically converted to images for processing
- Automatic photo extraction from the document
- Face detection identifies the main photo on the ID

### 3. Face Comparison
- Extracts face descriptors from both images
- Calculates similarity using Euclidean distance
- Returns similarity score (0-100%)

## Technical Details

### Face Recognition Models Used
- `tinyFaceDetector`: Fast face detection
- `faceLandmark68Net`: Facial landmark detection
- `faceRecognitionNet`: Face descriptor extraction

### Client-Side Processing
- All face detection and comparison happens in the browser
- Uses face-api.js models loaded from `/public/models/`
- No server-side processing required for face comparison

### Similarity Scoring
- **High Match**: >60% similarity
- **Possible Match**: 40-60% similarity  
- **Low Match**: <40% similarity

## Usage Notes

- Ensure good lighting for accurate face detection
- Position face clearly in camera frame
- Upload clear, high-quality ID documents
- Results may vary based on image quality and lighting conditions

## File Structure

```
src/
├── components/
│   └── PreprodFaceComparison.jsx    # Main testing component
└── pages/
    └── preprod.js                    # Preprod page route
```

## Dependencies

- `face-api.js`: Face detection and recognition
- `react-webcam`: Webcam access
- `framer-motion`: Animations
- `pdfjs-dist`: PDF to image conversion

## Testing

This component is designed for testing purposes and should not be used in production without proper security measures and validation.
