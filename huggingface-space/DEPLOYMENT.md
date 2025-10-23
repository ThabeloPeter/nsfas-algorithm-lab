# HuggingFace Space Deployment Guide

## Step-by-Step Deployment

### 1. Create HuggingFace Account
- Go to https://huggingface.co/join
- Sign up for a free account

### 2. Create a New Space
1. Go to https://huggingface.co/new-space
2. Fill in the details:
   - **Space name**: `nsfas-face-extraction` (or your preferred name)
   - **License**: MIT
   - **Select the Space SDK**: Docker
   - **Space hardware**: CPU basic (free tier)
   - **Visibility**: Public or Private (your choice)
3. Click "Create Space"

### 3. Upload Files
Upload all files from the `huggingface-space/` folder:
- `app.py`
- `requirements.txt`
- `Dockerfile`
- `README.md`
- `.gitignore`

You can do this via:
- **Web Interface**: Drag and drop files on the Space page
- **Git**: Clone the Space repo and push files
  ```bash
  git clone https://huggingface.co/spaces/YOUR_USERNAME/nsfas-face-extraction
  cd nsfas-face-extraction
  # Copy files from huggingface-space/ folder
  git add .
  git commit -m "Initial deployment"
  git push
  ```

### 4. Wait for Build
- HuggingFace will automatically build your Docker container
- This may take 5-10 minutes for the first build
- Watch the build logs on your Space page

### 5. Test the API
Once deployed, your API will be available at:
```
https://YOUR_USERNAME-nsfas-face-extraction.hf.space
```

Test the health endpoint:
```bash
curl https://YOUR_USERNAME-nsfas-face-extraction.hf.space/
```

### 6. Configure Frontend

Create a `.env.local` file in your Next.js project root:

```bash
# .env.local
NEXT_PUBLIC_HF_FACE_API_URL=https://YOUR_USERNAME-nsfas-face-extraction.hf.space
```

Replace `YOUR_USERNAME` with your actual HuggingFace username.

### 7. Restart Your Next.js Dev Server

```bash
npm run dev
```

## Local Testing (Before Deployment)

### Option 1: Using Docker
```bash
cd huggingface-space
docker build -t face-extraction-api .
docker run -p 7860:7860 face-extraction-api
```

### Option 2: Using Python Directly
```bash
cd huggingface-space
pip install -r requirements.txt
python app.py
```

Then in your `.env.local`:
```
NEXT_PUBLIC_HF_FACE_API_URL=http://localhost:7860
```

## API Documentation

Once deployed, visit:
```
https://YOUR_USERNAME-nsfas-face-extraction.hf.space/docs
```

For interactive API documentation (Swagger UI).

## Troubleshooting

### Build Fails
- Check the build logs on HuggingFace
- Ensure all dependencies in `requirements.txt` are compatible
- Verify Docker syntax in `Dockerfile`

### API Not Responding
- Check if the Space is running (green status on HF)
- Verify the URL is correct in `.env.local`
- Check CORS settings if getting CORS errors

### Slow Performance
- Free tier CPU can be slow for large images
- Consider upgrading to GPU hardware if needed
- Optimize image size before sending to API

## Upgrading to GPU (Optional)

For faster processing:
1. Go to your Space settings
2. Change Hardware from "CPU basic" to "T4 small" or similar
3. Note: GPU tiers are paid (~$0.60/hour, can pause when not in use)

## Monitoring

- View API logs on your HuggingFace Space page
- Monitor usage and performance
- Set up alerts for downtime (if needed)

## Cost

- **Free Tier**: CPU basic (always free, may be slower)
- **Paid Tier**: GPU access for faster processing
  - Can pause Space when not in use
  - Pay only for active time

## Security

Current setup allows all CORS origins for testing. For production:

1. Edit `app.py` line with `allow_origins`:
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://your-production-domain.com"],  # Specify your domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

2. Redeploy the Space

## Support

If you encounter issues:
- Check HuggingFace Spaces documentation: https://huggingface.co/docs/hub/spaces
- Review FastAPI docs: https://fastapi.tiangolo.com/
- Check the API logs on your Space page


