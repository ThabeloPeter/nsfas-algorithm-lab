# NSFAS Biometrics Verification

A focused biometrics verification platform for secure identity capture, face extraction, and selfie-to-ID matching. The project is designed to serve as a reusable building block that can plug into larger service journeys.

## What it does

- Capture ID documents using the browser camera
- Extract the primary face from Smart ID cards and Green ID books
- Run OCR on the source document
- Compare a live selfie against the extracted ID face
- Return metadata that can be used in downstream verification workflows

## Architecture

### Frontend
- Location: `src/`
- Built with Next.js and React
- Presents the capture and verification workflow
- Connects to the API through a small browser client

### Backend API
- Location: `huggingface-space/`
- Built with FastAPI
- Handles face detection, selection, extraction, OCR, and comparison
- Deployed separately on Hugging Face Spaces

## Key routes

- `/` is the main biometrics platform
- `/preprod` is a legacy alias that redirects to `/`
- `/preprod/face-extraction` is a legacy alias that redirects to `/`

## Tech Stack

- Frontend: Next.js 14, React 18, Tailwind CSS, Framer Motion, Lucide Icons
- Backend: FastAPI, OpenCV, MTCNN, PaddleOCR, `face_recognition`
- Deployment: Vercel for the frontend, Hugging Face Spaces for the API

## Environment

Create a `.env.local` file:

```bash
NEXT_PUBLIC_HF_FACE_API_URL=https://your-huggingface-space.hf.space
```

## Run locally

```bash
npm install
npm run dev
```

## Goal

This repo is now oriented around one product: biometrics verification. The intent is to provide a clean, modular foundation that can integrate with larger service journeys instead of a multi-tool lab.
