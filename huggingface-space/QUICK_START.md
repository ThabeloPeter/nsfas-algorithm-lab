# Quick Start Guide - Face Extraction API

Get your face extraction API running in under 10 minutes!

## Prerequisites

- HuggingFace account (free): https://huggingface.co/join
- Git (optional, for command-line deployment)

## Option 1: Deploy to HuggingFace (Recommended - 5 minutes)

### Step 1: Create Space (2 minutes)
1. Go to https://huggingface.co/new-space
2. Fill in:
   - Name: `nsfas-face-extraction`
   - SDK: **Docker**
   - Hardware: **CPU basic** (free)
3. Click **Create Space**

### Step 2: Upload Files (2 minutes)
Drag and drop these files to your Space:
- ✅ `app.py`
- ✅ `requirements.txt`
- ✅ `Dockerfile`
- ✅ `README.md`

### Step 3: Wait for Build (3-5 minutes)
Watch the build logs. When done, you'll see:
```
Running on http://0.0.0.0:7860
```

### Step 4: Get Your API URL
Your API is now live at:
```
https://YOUR_USERNAME-nsfas-face-extraction.hf.space
```

Replace `YOUR_USERNAME` with your HuggingFace username.

### Step 5: Test It
Visit:
```
https://YOUR_USERNAME-nsfas-face-extraction.hf.space/docs
```

You should see the interactive API documentation!

## Option 2: Run Locally (For Testing - 5 minutes)

### Prerequisites
- Python 3.9+
- pip

### Steps
```bash
# Navigate to the huggingface-space folder
cd huggingface-space

# Install dependencies (may take 3-5 minutes)
pip install -r requirements.txt

# Run the API
python app.py
```

API will be available at: `http://localhost:7860`

## Configure Your Frontend

### Step 1: Create Environment File
In your Next.js project root, create `.env.local`:

```bash
# Windows PowerShell
New-Item -Path .env.local -ItemType File

# Linux/Mac
touch .env.local
```

### Step 2: Add API URL
Edit `.env.local`:

```env
# For HuggingFace Space
NEXT_PUBLIC_HF_FACE_API_URL=https://YOUR_USERNAME-nsfas-face-extraction.hf.space

# OR for local testing
# NEXT_PUBLIC_HF_FACE_API_URL=http://localhost:7860
```

### Step 3: Restart Next.js
```bash
# Stop current server (Ctrl+C)
# Then restart
npm run dev
```

## Test Your Setup

1. Go to: `http://localhost:3000/preprod/face-extraction`
2. Upload an ID document (JPEG, PNG, or PDF)
3. Watch it extract the face!

## Troubleshooting

### "Unable to connect to face extraction service"
✅ Check `.env.local` file exists  
✅ Verify API URL is correct  
✅ Restart Next.js dev server  
✅ Check HuggingFace Space is running (green status)

### "No face detected"
✅ Ensure ID document has a clear visible face  
✅ Try uploading as JPEG instead of PDF  
✅ Check image quality (not too dark/blurry)

### Build fails with "Exit code 137" (Out of Memory)
✅ **FIXED**: Updated Dockerfile now uses pre-built image  
✅ Re-upload the updated `Dockerfile` from this folder  
✅ See `BUILD_TROUBLESHOOTING.md` for details  
✅ Alternative: Temporarily upgrade to CPU upgrade (~$0.01 cost for build)

### Build fails on HuggingFace (other errors)
✅ Check build logs for specific error  
✅ Verify all files were uploaded correctly  
✅ Ensure README.md has the header configuration  
✅ Wait a bit longer (first build can take 10-15 minutes)

## Next Steps

- 📖 Read `DEPLOYMENT.md` for advanced configuration
- 🔧 Check `README.md` for API documentation
- 🎨 Customize the API for your needs
- 📊 Monitor usage on your HuggingFace Space page

## Cost

- **Free Tier (CPU basic)**: $0/month, always free
- **Paid Tier (GPU)**: ~$0.60/hour (optional, for faster processing)

## Support

Need help?
- HuggingFace Docs: https://huggingface.co/docs/hub/spaces
- FastAPI Docs: https://fastapi.tiangolo.com/
- Check API logs on your Space page

---

**Congratulations! Your face extraction API is ready!** 🎉

