# Preprod Testing Environment - Structure Guide

## 🎯 Overview

The preprod environment has been restructured into a **menu-based testing hub** where all testing tools are organized in one place. This makes it easy to add, manage, and delete testing components without affecting the main application.

## 📂 Directory Structure

```
nsfas-profile-reset/
│
├── src/
│   ├── preprod/                          # 🧪 ALL PREPROD CODE (isolated)
│   │   ├── components/                   # Preprod components
│   │   │   ├── PreprodMenu.jsx          # Main dashboard/menu
│   │   │   └── FaceExtractionTool.jsx   # Face extraction tool
│   │   └── README.md                    # Preprod documentation
│   │
│   └── pages/
│       ├── preprod.js                    # Main preprod entry (/preprod)
│       └── preprod/                      # Preprod routes
│           └── face-extraction.js        # Face extraction page (/preprod/face-extraction)
│
└── PREPROD_STRUCTURE.md                  # This file
```

## 🚀 How It Works

### 1. Main Entry Point: `/preprod`
- Shows a menu/dashboard with all available testing tools
- Displays tool cards with descriptions
- Provides easy navigation to each tool
- Located at: `src/pages/preprod.js`

### 2. Individual Tools: `/preprod/[tool-name]`
- Each tool has its own route
- Example: `/preprod/face-extraction`
- Located in: `src/pages/preprod/[tool-name].js`

### 3. Component Organization
- All preprod components are in `src/preprod/components/`
- Each tool is a separate component
- The menu component (`PreprodMenu.jsx`) manages navigation

## 🎨 Navigation Flow

```
User visits /preprod
    ↓
PreprodMenu displays all tools
    ↓
User clicks "Face Extraction Tool"
    ↓
Navigates to /preprod/face-extraction
    ↓
FaceExtractionTool component loads
    ↓
User can click "Back to Preprod Hub" to return
```

## ➕ Adding a New Testing Tool

Follow these steps to add a new testing tool:

### Step 1: Create the Component
Create your tool component in `src/preprod/components/YourTool.jsx`:

```javascript
'use client';

export default function YourTool() {
  return (
    <div className="min-h-screen bg-gray-50 py-8 px-4">
      <div className="max-w-5xl mx-auto">
        <h1 className="text-3xl font-bold text-gray-900 mb-8">
          Your Tool Name
        </h1>
        {/* Your tool UI here */}
      </div>
    </div>
  );
}
```

### Step 2: Create the Page Route
Create a page in `src/pages/preprod/your-tool.js`:

```javascript
import Head from 'next/head';
import dynamic from 'next/dynamic';
import Link from 'next/link';
import { ArrowLeft } from 'lucide-react';

const YourTool = dynamic(() => import('@/preprod/components/YourTool'), { 
  ssr: false,
  loading: () => (
    <div className="min-h-screen bg-gray-50 flex items-center justify-center">
      <div className="text-center">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-red-600 mx-auto mb-4"></div>
        <p className="text-gray-600">Loading Tool...</p>
      </div>
    </div>
  )
});

export default function YourToolPage() {
  return (
    <>
      <Head>
        <title>Your Tool - Preprod</title>
      </Head>
      
      <div className="bg-white border-b shadow-sm sticky top-0 z-10">
        <div className="max-w-7xl mx-auto px-4 py-3">
          <Link 
            href="/preprod"
            className="inline-flex items-center space-x-2 px-3 py-2 text-gray-600 hover:text-gray-900 hover:bg-gray-100 rounded-lg transition-colors text-sm font-medium"
          >
            <ArrowLeft className="w-4 h-4" />
            <span>Back to Preprod Hub</span>
          </Link>
        </div>
      </div>

      <YourTool />
    </>
  );
}
```

### Step 3: Add to Menu
Edit `src/preprod/components/PreprodMenu.jsx` and add your tool to the `testTools` array:

```javascript
import { YourIcon } from 'lucide-react'; // Choose an icon

const testTools = [
  // ... existing tools
  {
    id: 'your-tool',
    title: 'Your Tool Name',
    description: 'Brief description of what your tool does',
    icon: YourIcon,
    color: 'blue', // red, blue, green, purple, etc.
    href: '/preprod/your-tool',
    status: 'active'
  }
];
```

### Step 4: Test
1. Run `npm run dev`
2. Visit `http://localhost:3000/preprod`
3. Click on your new tool card
4. Test functionality

## 🗑️ Deleting All Preprod Code

To completely remove the preprod environment:

```bash
# Delete preprod folder
rm -rf src/preprod/

# Delete preprod pages
rm -rf src/pages/preprod/
rm src/pages/preprod.js

# Delete this documentation (optional)
rm PREPROD_STRUCTURE.md
rm PREPROD_README.md
rm PREPROD_IMPLEMENTATION.md
```

**This will NOT affect your main application!** All preprod code is isolated.

## 🛠️ Current Testing Tools

### 1. Face Extraction Tool
- **Route**: `/preprod/face-extraction`
- **Component**: `src/preprod/components/FaceExtractionTool.jsx`
- **Purpose**: Test face detection and extraction from ID documents
- **Features**:
  - Upload ID documents (JPEG, PNG, PDF)
  - Automatic PDF to image conversion
  - Face detection with face-api.js
  - Smart face selection algorithm
  - High-quality extraction
  - Detailed metadata display
  - Download extracted faces

## 📝 File Locations Summary

| Purpose | File Path | URL |
|---------|-----------|-----|
| Main preprod entry | `src/pages/preprod.js` | `/preprod` |
| Preprod menu | `src/preprod/components/PreprodMenu.jsx` | - |
| Face extraction page | `src/pages/preprod/face-extraction.js` | `/preprod/face-extraction` |
| Face extraction tool | `src/preprod/components/FaceExtractionTool.jsx` | - |
| Preprod README | `src/preprod/README.md` | - |

## 🎯 Benefits of This Structure

1. **Isolated**: All preprod code in one folder - easy to find and delete
2. **Organized**: Menu-based navigation - professional and clean
3. **Scalable**: Easy to add new tools - just 3 simple steps
4. **Safe**: No interference with main app - completely separate
5. **Professional**: Looks like a real testing hub - not just random pages
6. **Documented**: Clear structure and instructions - easy for team members

## ⚠️ Important Notes

- All preprod components should use `'use client'` directive
- Components using face-api.js need `ssr: false` in dynamic import
- Keep preprod code isolated - don't import into main app
- This is for testing only - not production code
- Clean up unused tools regularly to avoid clutter

## 🔍 Quick Reference

```bash
# Access preprod hub
http://localhost:3000/preprod

# Access specific tool
http://localhost:3000/preprod/[tool-name]

# Example: Face extraction
http://localhost:3000/preprod/face-extraction
```

---

**Last Updated**: October 22, 2025  
**Version**: 1.0.0


