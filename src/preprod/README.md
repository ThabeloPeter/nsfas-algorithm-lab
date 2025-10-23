# Preprod Testing Environment

This directory contains isolated testing components and tools for internal development and feature testing.

## 📁 Structure

```
src/preprod/
├── components/          # Reusable preprod components
│   ├── PreprodMenu.jsx      # Main menu/dashboard
│   └── FaceExtractionTool.jsx  # Face extraction testing tool (API-based)
├── ENV_SETUP.md        # Environment variable setup guide
├── VISION_API_INTEGRATION.md  # Vision API integration options
└── README.md           # This file
```

## 🚀 Accessing Preprod

- **Main Hub**: `/preprod` - Menu-based dashboard to access all testing tools
- **Face Extraction**: `/preprod/face-extraction` - ID face extraction testing tool

## 🛠️ Available Tools

### 1. Face Extraction Tool (API-Based)
- **Path**: `/preprod/face-extraction`
- **Purpose**: Test face detection and extraction from ID documents
- **Technology**: Server-side processing via HuggingFace Spaces API
- **Features**:
  - Upload JPEG, PNG, or PDF ID documents
  - Automatic PDF to image conversion (server-side)
  - Face detection using Python `face_recognition` library
  - Smart face selection (70% size + 30% position scoring)
  - High-quality face extraction
  - Detailed extraction metadata
  - Download extracted faces
  - Face comparison capability (selfie vs ID photo)

#### Why API-Based?
The previous client-side approach using face-api.js was causing:
- ❌ High RAM usage (~2GB+)
- ❌ Browser crashes on large files
- ❌ Slow model loading

The new server-side API provides:
- ✅ Minimal browser RAM usage
- ✅ No crashes
- ✅ Faster processing
- ✅ More accurate detection
- ✅ Free hosting on HuggingFace Spaces

## 🔧 Setup Requirements

### 1. Environment Configuration
Create a `.env.local` file in your project root:

```env
# For deployed HuggingFace Space (recommended)
NEXT_PUBLIC_HF_FACE_API_URL=https://your-username-face-extraction.hf.space

# Or for local API testing
# NEXT_PUBLIC_HF_FACE_API_URL=http://localhost:7860
```

See `ENV_SETUP.md` for detailed instructions.

### 2. Deploy HuggingFace API (One-time Setup)
The face extraction tool requires a backend API. See `../../../huggingface-space/DEPLOYMENT.md` for:
- Step-by-step HuggingFace Spaces deployment
- Local API testing instructions
- API documentation

## ➕ Adding New Testing Tools

1. Create your component in `src/preprod/components/YourTool.jsx`
2. Create a page in `src/pages/preprod/your-tool.js` that dynamically imports your component
3. Add your tool to the menu in `src/preprod/components/PreprodMenu.jsx`:

```javascript
{
  id: 'your-tool',
  title: 'Your Tool Name',
  description: 'Description of what it does',
  icon: YourIcon, // from lucide-react
  href: '/preprod/your-tool',
  status: 'active',
  tag: 'Active'
}
```

## 🗑️ Cleanup

To remove all preprod components:

1. Delete the entire `src/preprod/` folder
2. Delete `src/pages/preprod/` folder
3. Delete `src/pages/preprod.js` file
4. Delete `src/lib/faceExtractionApi.js` file
5. Delete `huggingface-space/` folder (if API no longer needed)
6. Remove `NEXT_PUBLIC_HF_FACE_API_URL` from `.env.local` and deployment configs

This will not affect the main application as all preprod code is isolated.

## ⚠️ Important Notes

- All preprod components use `'use client'` directive for client-side rendering
- Face extraction now uses server-side API (HuggingFace Spaces)
- This environment is for **testing only** - do not use in production
- Changes here do not affect the main NSFAS application
- API calls are made directly from the browser to HuggingFace (CORS enabled)

## 🔧 Dependencies

### Frontend
- `lucide-react` - Icons
- `framer-motion` - Animations
- Custom API client (`src/lib/faceExtractionApi.js`)

### Backend (HuggingFace Space)
- `FastAPI` - Web framework
- `face_recognition` - Face detection and recognition
- `opencv-python-headless` - Image processing
- `PyMuPDF` - PDF processing
- `Pillow` - Image manipulation

## 📝 Architecture

### Previous (Client-Side)
```
User Browser
  ↓
Upload ID → Load face-api.js models (heavy) → Detect faces → Extract → Display
  ↓
High RAM usage, crashes
```

### Current (Server-Side API)
```
User Browser
  ↓
Upload ID → Send to HuggingFace API → Python processes image → Return face
  ↓
Minimal RAM usage, fast, reliable
```

## 🚀 Quick Start

1. **First-time setup**:
   ```bash
   # Deploy API to HuggingFace (see huggingface-space/DEPLOYMENT.md)
   # Then create .env.local with your API URL
   ```

2. **Run development server**:
   ```bash
   npm run dev
   ```

3. **Access tools**:
   ```
   http://localhost:3000/preprod
   ```

## 📚 Documentation

- `ENV_SETUP.md` - How to configure environment variables
- `VISION_API_INTEGRATION.md` - Alternative vision API options
- `../../../huggingface-space/DEPLOYMENT.md` - API deployment guide
- `../../../huggingface-space/README.md` - API documentation

## 🔒 Security Notes

- API currently allows all CORS origins (for testing)
- For production: Update CORS settings in `app.py` to specific domain
- Environment variables should never be committed to git
- `.env.local` is automatically gitignored

## 💡 Best Practices

1. Keep all preprod code isolated in this directory
2. Use descriptive names for tools and components
3. Add documentation for complex testing tools
4. Clean up unused tools regularly
5. Test API connectivity before deploying frontend
6. Monitor HuggingFace Space logs for API errors

---

**Last Updated**: October 2025  
**Architecture**: Server-Side API (HuggingFace Spaces)  
**Status**: Active
