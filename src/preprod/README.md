# Biometrics Verification UI

This directory now holds the biometrics verification experience and its legacy route aliases.

## Main flow

- `/` is the primary landing page
- `/preprod` redirects to `/`
- `/preprod/face-extraction` redirects to `/`

## Components

- `FaceExtractionTool.jsx` - Camera capture, face extraction, OCR, and selfie verification

## Backend

The UI talks to the FastAPI service in `huggingface-space/` through `src/lib/faceExtractionApi.js`.

## Environment

Set `NEXT_PUBLIC_HF_FACE_API_URL` in `.env.local` to point at the deployed API.
