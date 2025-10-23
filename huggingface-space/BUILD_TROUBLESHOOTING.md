# Build Troubleshooting - Exit Code 137

## Problem: Out of Memory Error (Exit Code 137)

Exit code 137 means the Docker build ran out of memory. This is common with `face_recognition` which requires building `dlib` from source.

## Solution 1: Use Updated Dockerfile (Recommended)

I've updated the Dockerfile to use a pre-built image that already has `face_recognition` installed. This avoids the memory-intensive build process.

**The updated Dockerfile now uses**:
```dockerfile
FROM animcogn/face_recognition:cpu
```

This pre-built image already includes:
- Python 3.8+
- face_recognition library
- dlib
- OpenCV
- NumPy

We only install the lightweight dependencies:
- FastAPI
- Uvicorn
- PyMuPDF
- python-multipart

**Action**: Re-upload the updated `Dockerfile` to your HuggingFace Space.

## Solution 2: Upgrade Hardware (If Solution 1 Fails)

If the pre-built image still fails, you can temporarily upgrade the hardware:

### Steps:
1. Go to your HuggingFace Space settings
2. Click on **"Settings"** tab
3. Under **"Hardware"**, change from:
   - ❌ CPU basic (free, 2GB RAM)
   - ✅ CPU upgrade (~$0.05/hour, 8GB RAM)
4. Wait for rebuild
5. Once built successfully, you can downgrade back to CPU basic
6. The built image will be cached and won't need to rebuild

### Cost:
- Build time: ~5-10 minutes
- Cost: ~$0.01-0.02 for the build
- After build: Downgrade back to free tier

## Solution 3: Local Build & Push (Advanced)

If you have Docker installed locally:

```bash
cd huggingface-space

# Build locally (requires 4GB+ RAM)
docker build -t face-extraction-api .

# Tag for HuggingFace
docker tag face-extraction-api registry.hf.space/YOUR_USERNAME/nsfas-face-extraction:latest

# Login to HuggingFace registry
huggingface-cli login

# Push to HuggingFace
docker push registry.hf.space/YOUR_USERNAME/nsfas-face-extraction:latest
```

## Solution 4: Use Requirements.txt Approach (Not Docker)

If Docker continues to fail, you can change your Space to use Python SDK instead:

### Steps:
1. In HuggingFace Space settings, change SDK from **Docker** to **Gradio** or **Streamlit**
2. Upload only these files:
   - `app.py`
   - `requirements.txt`
3. Update README.md header:
   ```yaml
   ---
   title: NSFAS Face Extraction API
   emoji: 🔍
   colorFrom: blue
   colorTo: indigo
   sdk: gradio
   sdk_version: "4.0.0"
   app_file: app.py
   pinned: false
   ---
   ```
4. Modify `app.py` to use Gradio interface instead of FastAPI

**Note**: This requires rewriting the API as a Gradio app. Let me know if you want me to create this version.

## Recommended Approach

**Try in this order**:

1. ✅ **Use updated Dockerfile** (pre-built image) - Try this first!
2. If fails → **Upgrade hardware temporarily** ($0.01-0.02 cost)
3. If still fails → **Use local build & push**
4. Last resort → **Switch to Gradio SDK**

## Updated Files

The Dockerfile has been updated to use the lightweight pre-built image. Simply:

1. Delete your current HuggingFace Space (or create a new one)
2. Upload the updated files:
   - ✅ Updated `Dockerfile` (now uses pre-built image)
   - ✅ `app.py`
   - ✅ `requirements.txt`
   - ✅ `README.md`

## Verification

After successful build, you should see:
```
Running on http://0.0.0.0:7860
Application startup complete.
```

Then test at:
```
https://YOUR_USERNAME-nsfas-face-extraction.hf.space/docs
```

## Still Having Issues?

If the updated Dockerfile still fails, let me know and I'll create a Gradio-based version which is guaranteed to work on the free tier.

---

**Updated**: October 2025  
**Status**: Dockerfile optimized for low-memory build


