# Deployment Guide - NSFAS Algorithm Lab

## Architecture Overview

```
Frontend (Vercel) ←→ Backend API (HuggingFace Space)
```

## Step-by-Step Deployment

### 1️⃣ Deploy Backend API to HuggingFace Space

#### Create the Space
1. Go to https://huggingface.co/new-space
2. Fill in:
   - **Space name**: `nsfas-face-extraction`
   - **SDK**: Select **Docker**
   - **Visibility**: Public
3. Click **Create Space**

#### Upload Files
Upload these files from the `huggingface-space/` folder:
- `app.py`
- `requirements.txt`
- `Dockerfile` (if present)
- `README.md`

#### Wait for Build
- Build time: ~5-10 minutes
- Status shown at top of Space page
- Green checkmark = Ready!

#### Get Your API URL
Your Space URL will be:
```
https://thabelotp-nsfas-face-extraction.hf.space
```

Format: `https://[username]-[space-name].hf.space`

#### Test the API
Open in browser:
```
https://thabelotp-nsfas-face-extraction.hf.space
```

You should see:
```json
{
  "status": "healthy",
  "service": "NSFAS Face Extraction API - OpenCV",
  "version": "2.0.0",
  "detector": "OpenCV Haar Cascade"
}
```

---

### 2️⃣ Deploy Frontend to Vercel

#### Already Deployed?
Your frontend is already on Vercel at:
```
https://nsfas-algorithm-lab.vercel.app
```

#### Configure Environment Variable
1. Go to https://vercel.com/dashboard
2. Select your project: `nsfas-algorithm-lab`
3. Click **Settings** → **Environment Variables**
4. Add new variable:
   - **Name**: `NEXT_PUBLIC_HF_FACE_API_URL`
   - **Value**: `https://thabelotp-nsfas-face-extraction.hf.space`
   - **Environments**: Select all (Production, Preview, Development)
5. Click **Save**

#### Redeploy
1. Go to **Deployments** tab
2. Click **⋯** (three dots) on latest deployment
3. Click **Redeploy**
4. Wait ~2-3 minutes

---

### 3️⃣ Local Development Setup

#### Create Local Environment File
```bash
cp .env.example .env.local
```

Edit `.env.local`:
```env
NEXT_PUBLIC_HF_FACE_API_URL=https://thabelotp-nsfas-face-extraction.hf.space
```

#### Run Locally
```bash
npm run dev
```

Visit: http://localhost:3000

---

## Troubleshooting

### ❌ Error: ERR_NAME_NOT_RESOLVED

**Problem**: API URL not set or incorrect format

**Solution**:
1. Check HuggingFace Space is deployed and running
2. Verify URL format: `https://username-spacename.hf.space`
3. Update Vercel environment variable
4. Redeploy Vercel

### ❌ Error: 404 Not Found

**Problem**: HuggingFace Space not deployed

**Solution**: Complete Step 1 (Deploy to HuggingFace)

### ❌ Error: CORS

**Problem**: API blocking requests

**Solution**: Already configured in `app.py` with:
```python
allow_origins=["*"]
```

### ❌ Slow API Response

**Problem**: Cold start on HuggingFace Space

**Solution**: 
- First request may take 20-30 seconds (waking up)
- Subsequent requests: 2-3 seconds
- Consider upgrading to persistent Space

---

## Environment Variables Reference

| Variable | Description | Example |
|----------|-------------|---------|
| `NEXT_PUBLIC_HF_FACE_API_URL` | HuggingFace Space URL | `https://thabelotp-nsfas-face-extraction.hf.space` |

**Note**: Must start with `NEXT_PUBLIC_` to be accessible in browser!

---

## Testing Your Deployment

### Test Backend API
```bash
curl https://thabelotp-nsfas-face-extraction.hf.space
```

Expected response:
```json
{"status": "healthy", "service": "NSFAS Face Extraction API - OpenCV"}
```

### Test Frontend
1. Visit: https://nsfas-algorithm-lab.vercel.app
2. Navigate to Algorithm Lab
3. Click "Face Extraction"
4. Upload an ID document
5. Should see extracted face in ~3 seconds

---

## URLs Summary

| Service | URL |
|---------|-----|
| Frontend | https://nsfas-algorithm-lab.vercel.app |
| Backend API | https://thabelotp-nsfas-face-extraction.hf.space |
| GitHub Repo | https://github.com/ThabeloPeter/nsfas-algorithm-lab |

---

## Next Steps After Deployment

1. ✅ Test face extraction with real ID documents
2. ✅ Monitor HuggingFace Space logs for errors
3. ✅ Add more algorithms to the lab
4. ✅ Share your Algorithm Lab URL!

---

## Support

- **HuggingFace Spaces**: https://huggingface.co/docs/hub/spaces
- **Vercel Docs**: https://vercel.com/docs
- **Next.js Docs**: https://nextjs.org/docs

---

**Ready to deploy? Start with Step 1! 🚀**

