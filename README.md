# NSFAS Algorithm Lab

A showcase and testing environment for innovative algorithms and solutions designed for NSFAS. This platform demonstrates cutting-edge approaches to real-world challenges including face extraction, verification, and other machine learning solutions.

## Architecture

### Backend API (Hugging Face Space)
- **Location**: `huggingface-space/`
- Python-based face extraction API using FastAPI
- Deployed separately on Hugging Face Spaces
- Handles face detection, extraction, and comparison
- See `huggingface-space/README.md` for details

### Frontend (Next.js)
- **Location**: `src/`
- React-based web interface
- Deployed on Vercel
- Connects to HuggingFace Space API
- Modern UI with Tailwind CSS + Framer Motion

## Tech Stack

- **Frontend**: Next.js 14, React 18, Tailwind CSS, Framer Motion, Lucide Icons
- **Backend**: FastAPI, Face Recognition, OpenCV, PyMuPDF
- **Deployment**: Vercel (Frontend) + HuggingFace Spaces (Backend API)

## Documentation

- `FACE_EXTRACTION_API_MIGRATION.md` - API migration guide
- `FINAL_SETUP_INSTRUCTIONS.md` - Setup instructions
- `NEXT_STEPS.md` - Development roadmap
- `PREPROD_CHANGELOG.md` - Change history
- `PREPROD_IMPLEMENTATION.md` - Implementation details
- `PREPROD_README.md` - Original preprod documentation
- `PREPROD_STRUCTURE.md` - Project structure
- `PREPROD_THEME_GUIDE.md` - Theme and styling guide

## Quick Start

### Install Dependencies
```bash
npm install
```

### Development
```bash
npm run dev
```

### Environment Variables
Create a `.env.local` file:
```
NEXT_PUBLIC_HF_FACE_API_URL=https://your-huggingface-space.hf.space
```

### Deployment
- Frontend deploys automatically to Vercel via GitHub
- Backend API runs on HuggingFace Space

See `FINAL_SETUP_INSTRUCTIONS.md` for detailed setup instructions.

## License

See LICENSE file in the main NSFAS repository.

