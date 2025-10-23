---
title: NSFAS Face Extraction API
emoji: 🔍
colorFrom: blue
colorTo: indigo
sdk: docker
app_file: app.py
pinned: false
---

# NSFAS Face Extraction API

A FastAPI service for extracting and analyzing faces from ID documents (images and PDFs).

## Features

- ✅ Face detection from JPEG, PNG, and PDF files
- ✅ Intelligent face selection (prioritizes larger, centered faces)
- ✅ Face extraction with metadata
- ✅ Face comparison between two images
- ✅ CORS enabled for web applications
- ✅ Fast and accurate using `face_recognition` library

## API Endpoints

### 1. Health Check
```
GET /
```
Returns API status and version.

### 2. Extract Face
```
POST /extract-face
```
Extracts the most prominent face from an ID document.

**Request:**
- Method: `POST`
- Content-Type: `multipart/form-data`
- Body: `file` (JPEG, PNG, or PDF, max 10MB)

**Response:**
```json
{
  "success": true,
  "face_image": "data:image/jpeg;base64,...",
  "metadata": {
    "total_faces": 1,
    "selected_index": 0,
    "confidence": 87.5,
    "face_size": {
      "width": 345,
      "height": 412
    },
    "position_score": 78.3,
    "size_score": 92.1,
    "combined_score": 87.5,
    "all_faces": [...]
  }
}
```

### 3. Compare Faces
```
POST /compare-faces
```
Compares two face images (e.g., selfie vs ID photo).

**Request:**
- Method: `POST`
- Content-Type: `multipart/form-data`
- Body: 
  - `selfie` (image file)
  - `id_photo` (image file)

**Response:**
```json
{
  "success": true,
  "match": true,
  "distance": 0.42,
  "confidence": 58.0,
  "threshold": 0.6
}
```

## Algorithm

The face selection algorithm scores detected faces based on:
- **Size Score (70% weight)**: Larger faces are prioritized
- **Position Score (30% weight)**: Faces closer to the top-center are preferred
- **Combined Score**: Weighted average of both scores

This ensures the main ID photo face is selected over smaller watermark/security faces.

## Deployment

### HuggingFace Spaces

1. Create a new Space on [HuggingFace](https://huggingface.co/spaces)
2. Choose "Docker" as the SDK
3. Upload these files:
   - `app.py`
   - `requirements.txt`
   - `Dockerfile`
   - `README.md`
4. The Space will automatically build and deploy

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run the server
python app.py
# or
uvicorn app:app --reload --port 7860
```

Visit `http://localhost:7860/docs` for interactive API documentation.

## Usage Example

### JavaScript/Frontend

```javascript
const formData = new FormData();
formData.append('file', idDocumentFile);

const response = await fetch('https://your-space.hf.space/extract-face', {
  method: 'POST',
  body: formData
});

const data = await response.json();
console.log('Extracted face:', data.face_image);
console.log('Metadata:', data.metadata);
```

### Python

```python
import requests

with open('id_document.pdf', 'rb') as f:
    files = {'file': f}
    response = requests.post('https://your-space.hf.space/extract-face', files=files)
    data = response.json()
    print(data['metadata'])
```

## Error Handling

The API returns appropriate HTTP status codes:
- `200`: Success
- `400`: Bad request (invalid file, unsupported format, file too large)
- `404`: No face detected
- `500`: Internal server error

## Technology Stack

- **FastAPI**: Modern, fast web framework
- **face_recognition**: Robust face detection and recognition
- **OpenCV**: Image processing
- **PyMuPDF**: PDF handling
- **Pillow**: Image manipulation

## License

MIT License - Free for use in NSFAS projects.

## Author

NSFAS Development Team

