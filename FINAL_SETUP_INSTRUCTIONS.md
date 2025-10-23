# ✅ API is Running! Final Setup Steps

Your HuggingFace Space API is now successfully deployed and running! Just a few more steps to connect it to your frontend.

## 🎯 What You Need to Do

### Step 1: Find Your API URL

Your API is running at:
```
https://YOUR_USERNAME-nsfas-face-extraction.hf.space
```

**To find your exact URL:**
1. Go to your HuggingFace Space page
2. Look at the URL or the "Running on..." message
3. Copy the full URL (e.g., `https://johndoe-nsfas-face-extraction.hf.space`)

**Test your API:**
Visit: `https://YOUR_USERNAME-nsfas-face-extraction.hf.space/docs`

You should see the interactive API documentation (Swagger UI).

### Step 2: Create `.env.local` File

In your project root folder (`nsfas-profile-reset/`), create a file named `.env.local`

**Using PowerShell:**
```powershell
# Create the file
New-Item -Path .env.local -ItemType File -Force

# Add the configuration (replace with YOUR actual URL)
Add-Content -Path .env.local -Value "NEXT_PUBLIC_HF_FACE_API_URL=https://YOUR_USERNAME-nsfas-face-extraction.hf.space"
```

**Or manually:**
1. Right-click in the `nsfas-profile-reset` folder
2. New → Text Document
3. Rename to `.env.local` (note the dot at the beginning)
4. Open in Notepad and add:
```
NEXT_PUBLIC_HF_FACE_API_URL=https://YOUR_USERNAME-nsfas-face-extraction.hf.space
```
5. Save and close

**Important:** Replace `YOUR_USERNAME-nsfas-face-extraction` with your actual HuggingFace Space URL!

### Step 3: Verify `.env.local` Content

Your `.env.local` file should contain exactly one line:
```
NEXT_PUBLIC_HF_FACE_API_URL=https://your-actual-space-url.hf.space
```

**Common mistakes to avoid:**
- ❌ Don't add quotes around the URL
- ❌ Don't add trailing slash at the end
- ❌ Don't leave `YOUR_USERNAME` - replace it with actual username
- ✅ Do use your actual HuggingFace Space URL

### Step 4: Restart Development Server

**If server is running:**
1. Press `Ctrl + C` to stop it
2. Run: `npm run dev`

**The server will:**
- Load the new environment variable
- Connect to your HuggingFace API
- Be ready to test!

### Step 5: Test the Face Extraction Tool

1. Open browser: `http://localhost:3000/preprod/face-extraction`
2. Upload an ID document (JPEG, PNG, or PDF)
3. Watch it extract the face **without crashing!** 🎉

## ✅ Success Indicators

You'll know it's working when:
- ✅ Upload shows a loading spinner
- ✅ Face is extracted and displayed
- ✅ Metadata shows (faces detected, confidence, etc.)
- ✅ No browser crashes!
- ✅ Console shows: "✅ Face extraction successful"

## 🐛 Troubleshooting

### "Unable to connect to face extraction service"
**Cause:** Environment variable not loaded or incorrect URL

**Fix:**
1. Check `.env.local` file exists in project root
2. Verify URL is correct (no typos)
3. Ensure no trailing slash at end of URL
4. Restart dev server (`Ctrl+C` then `npm run dev`)
5. Check HuggingFace Space is still running (green status)

### "CORS Error" in console
**Cause:** API not allowing your domain

**Fix:** The API already allows all origins for testing. If you still get CORS errors:
1. Check you're using the correct API URL
2. Verify the API is running on HuggingFace
3. Try accessing `/docs` endpoint to confirm API is accessible

### Environment variable not working
**Verify it's loaded:**
1. Open browser console on `/preprod/face-extraction`
2. Type: `console.log(process.env.NEXT_PUBLIC_HF_FACE_API_URL)`
3. Should show your API URL
4. If shows `undefined`, restart dev server

### Still having issues?
1. Check HuggingFace Space logs for errors
2. Verify API responds at `/docs` endpoint
3. Check browser console for detailed error messages
4. Ensure dev server was restarted after creating `.env.local`

## 📊 Testing Checklist

Test these scenarios:
- [ ] Upload JPEG ID document
- [ ] Upload PNG ID document
- [ ] Upload PDF ID document
- [ ] Large file (5-10MB)
- [ ] ID with multiple faces (should select main one)
- [ ] Download extracted face
- [ ] Verify metadata is accurate

## 🚀 For Production (Later)

When deploying to Vercel:
1. Go to Vercel project settings
2. Environment Variables section
3. Add:
   - **Name:** `NEXT_PUBLIC_HF_FACE_API_URL`
   - **Value:** Your HuggingFace Space URL
   - **Environments:** Production, Preview, Development
4. Redeploy

## 📝 Summary

Current Status:
- ✅ HuggingFace API deployed and running
- ⏳ Frontend needs `.env.local` configuration
- ⏳ Need to restart dev server
- ⏳ Ready to test!

Next Actions:
1. Create `.env.local` with your API URL
2. Restart dev server
3. Test at `/preprod/face-extraction`
4. Enjoy crash-free face extraction! 🎉

---

**Need Help?**
- HuggingFace Space URL format: `https://username-space-name.hf.space`
- No trailing slash
- No quotes in `.env.local`
- Must restart dev server after creating `.env.local`

**Your API is ready! Just configure the environment variable and test!** 🚀


