# Vision API Integration Guide

This document outlines how to integrate various Vision APIs into the preprod testing environment.

## 🎯 Recommended APIs for ID Verification

### 1. Google Cloud Vision API
**Best for**: Face extraction + OCR (ID number, name, etc.)

#### Setup:
1. Create a Google Cloud project
2. Enable Vision API
3. Create service account and download credentials
4. Add to `.env.local`:
```env
GOOGLE_VISION_KEY=your_api_key_here
```

#### Implementation Example:
```javascript
// src/preprod/components/GoogleVisionTool.jsx
'use client';

import { useState } from 'react';

export default function GoogleVisionTool() {
  const [result, setResult] = useState(null);

  const analyzeId = async (file) => {
    const base64 = await fileToBase64(file);
    
    const response = await fetch('/api/preprod/google-vision', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ image: base64 })
    });
    
    const data = await response.json();
    setResult(data);
  };

  return (
    <div>
      {/* Upload UI */}
      {result && (
        <div>
          <h3>Faces Found: {result.faces.length}</h3>
          <h3>Text Detected:</h3>
          <p>{result.text}</p>
        </div>
      )}
    </div>
  );
}
```

```javascript
// src/pages/api/preprod/google-vision.js
export default async function handler(req, res) {
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  try {
    const { image } = req.body;
    
    const response = await fetch(
      `https://vision.googleapis.com/v1/images:annotate?key=${process.env.GOOGLE_VISION_KEY}`,
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          requests: [{
            image: { content: image },
            features: [
              { type: 'FACE_DETECTION', maxResults: 5 },
              { type: 'TEXT_DETECTION' },
              { type: 'DOCUMENT_TEXT_DETECTION' }
            ]
          }]
        })
      }
    );

    const data = await response.json();
    
    res.status(200).json({
      faces: data.responses[0].faceAnnotations || [],
      text: data.responses[0].fullTextAnnotation?.text || '',
      success: true
    });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
}
```

---

### 2. Microsoft Azure Computer Vision
**Best for**: High accuracy face detection + Smart cropping

#### Setup:
1. Create Azure account
2. Create Computer Vision resource
3. Get endpoint and subscription key
4. Add to `.env.local`:
```env
AZURE_VISION_ENDPOINT=https://your-resource.cognitiveservices.azure.com
AZURE_VISION_KEY=your_subscription_key
```

#### Implementation Example:
```javascript
// src/pages/api/preprod/azure-vision.js
export default async function handler(req, res) {
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  try {
    const { image } = req.body; // base64
    const imageBuffer = Buffer.from(image, 'base64');
    
    const endpoint = process.env.AZURE_VISION_ENDPOINT;
    const key = process.env.AZURE_VISION_KEY;
    
    const response = await fetch(
      `${endpoint}/vision/v3.2/analyze?visualFeatures=Faces&details=Landmarks`,
      {
        method: 'POST',
        headers: {
          'Ocp-Apim-Subscription-Key': key,
          'Content-Type': 'application/octet-stream'
        },
        body: imageBuffer
      }
    );

    const data = await response.json();
    
    res.status(200).json({
      faces: data.faces || [],
      success: true
    });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
}
```

---

### 3. Amazon Rekognition
**Best for**: Face comparison + Quality analysis

#### Setup:
1. Create AWS account
2. Create IAM user with Rekognition permissions
3. Get access key and secret
4. Add to `.env.local`:
```env
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_REGION=us-east-1
```

#### Implementation Example:
```javascript
// src/pages/api/preprod/aws-rekognition.js
import { RekognitionClient, DetectFacesCommand, CompareFacesCommand } from '@aws-sdk/client-rekognition';

const client = new RekognitionClient({
  region: process.env.AWS_REGION,
  credentials: {
    accessKeyId: process.env.AWS_ACCESS_KEY_ID,
    secretAccessKey: process.env.AWS_SECRET_ACCESS_KEY
  }
});

export default async function handler(req, res) {
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  try {
    const { image1, image2, action } = req.body; // base64 images
    
    if (action === 'compare') {
      // Compare two faces
      const command = new CompareFacesCommand({
        SourceImage: { Bytes: Buffer.from(image1, 'base64') },
        TargetImage: { Bytes: Buffer.from(image2, 'base64') },
        SimilarityThreshold: 80
      });
      
      const response = await client.send(command);
      
      res.status(200).json({
        matches: response.FaceMatches,
        similarity: response.FaceMatches[0]?.Similarity || 0,
        success: true
      });
    } else {
      // Detect faces
      const command = new DetectFacesCommand({
        Image: { Bytes: Buffer.from(image1, 'base64') },
        Attributes: ['ALL']
      });
      
      const response = await client.send(command);
      
      res.status(200).json({
        faces: response.FaceDetails,
        success: true
      });
    }
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
}
```

---

## 🎨 Adding Vision API Tool to Preprod

### Step 1: Install Dependencies
```bash
# For Google Cloud Vision
npm install @google-cloud/vision

# For Azure Computer Vision
npm install @azure/cognitiveservices-computervision

# For AWS Rekognition
npm install @aws-sdk/client-rekognition
```

### Step 2: Create API Endpoint
Create your API route in `src/pages/api/preprod/[your-api].js`

### Step 3: Create Component
Create your tool component in `src/preprod/components/[YourAPI]Tool.jsx`

### Step 4: Create Page Route
Create page in `src/pages/preprod/[your-api].js`

### Step 5: Add to Menu
Update `src/preprod/components/PreprodMenu.jsx`:

```javascript
{
  id: 'google-vision',
  title: 'Google Vision API',
  description: 'Test Google Cloud Vision for face detection and OCR',
  icon: Cloud, // from lucide-react
  color: 'blue',
  href: '/preprod/google-vision',
  status: 'active'
}
```

---

## 🔄 Hybrid Approach (Recommended)

Use multiple APIs for best results:

1. **Google Vision** - Extract text (ID number, name, dates)
2. **Azure Computer Vision** - Detect and extract face
3. **AWS Rekognition** - Compare selfie with extracted ID face
4. **Client-side face-api.js** - Fallback if APIs fail or for offline testing

### Example Flow:
```
1. User uploads ID document
   ↓
2. Google Vision extracts text (ID number, name)
   ↓
3. Azure Vision detects and extracts face
   ↓
4. User takes selfie
   ↓
5. AWS Rekognition compares selfie with ID face
   ↓
6. Return similarity score + extracted data
```

---

## 💰 Cost Comparison

| API | Free Tier | Pricing After Free Tier |
|-----|-----------|------------------------|
| Google Vision | 1,000 requests/month | $1.50 per 1,000 images |
| Azure Computer Vision | 5,000 transactions/month | $1.00 per 1,000 images |
| AWS Rekognition | 5,000 images/month (12 months) | $1.00 per 1,000 images |

**Recommendation**: Start with Azure (largest free tier) or AWS (if you have AWS account).

---

## 🔒 Security Best Practices

1. **Never commit API keys** - Use `.env.local`
2. **Add to `.gitignore`**:
```
.env.local
.env
```

3. **Use environment variables**:
```javascript
const apiKey = process.env.GOOGLE_VISION_KEY;
```

4. **API routes only** - Never expose keys to client-side
5. **Rate limiting** - Add rate limiting to API routes
6. **Input validation** - Validate file size and type

---

## 🧪 Testing Strategy

1. **Test with preprod environment first**
2. **Use dummy data** - Test with sample IDs
3. **Compare results** - Test same image with different APIs
4. **Measure accuracy** - Track false positives/negatives
5. **Monitor costs** - Watch API usage
6. **Performance testing** - Measure response times
7. **Error handling** - Test failure scenarios

---

## 📊 Example: Complete Vision API Testing Tool

Create a comprehensive testing tool that compares all APIs:

```javascript
// src/preprod/components/VisionAPIComparison.jsx
'use client';

import { useState } from 'react';

export default function VisionAPIComparison() {
  const [results, setResults] = useState({
    google: null,
    azure: null,
    aws: null,
    faceapi: null
  });

  const testAllAPIs = async (file) => {
    const base64 = await fileToBase64(file);
    
    // Test all APIs in parallel
    const [google, azure, aws, faceapi] = await Promise.all([
      testGoogleVision(base64),
      testAzureVision(base64),
      testAWSRekognition(base64),
      testFaceAPI(file)
    ]);
    
    setResults({ google, azure, aws, faceapi });
  };

  return (
    <div>
      <h1>Vision API Comparison</h1>
      {/* Upload UI */}
      
      {/* Results comparison table */}
      <table>
        <thead>
          <tr>
            <th>API</th>
            <th>Faces Found</th>
            <th>Processing Time</th>
            <th>Confidence</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>Google Vision</td>
            <td>{results.google?.faces || '-'}</td>
            <td>{results.google?.time || '-'}</td>
            <td>{results.google?.confidence || '-'}</td>
          </tr>
          {/* More rows... */}
        </tbody>
      </table>
    </div>
  );
}
```

---

## 📝 Next Steps

1. Choose your preferred Vision API
2. Set up account and get credentials
3. Create API endpoint in preprod
4. Create testing component
5. Add to preprod menu
6. Test thoroughly
7. Integrate into main app once validated

---

**Note**: All Vision API integrations should be tested in the preprod environment first before integrating into the main application.


