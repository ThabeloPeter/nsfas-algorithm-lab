# Preprod Terminal Theme Guide

## 🎨 Design System

The preprod environment uses a **terminal/hacker aesthetic** with JetBrains Mono font and a Matrix-inspired color scheme.

### Color Palette

```
Background:     #000000 (black)
Secondary BG:   #0f172a (slate-900)
Primary:        #4ade80 (green-400)
Accent:         #10b981 (emerald-500)
Secondary:      #22d3ee (cyan-400)
Warning:        #fbbf24 (yellow-400)
Error:          #f87171 (red-400)
```

### Typography

**Font Family**: JetBrains Mono (monospace)
- Imported from Google Fonts
- Weights: 400 (regular), 500 (medium), 700 (bold)

```css
font-family: 'JetBrains Mono', monospace
```

## 🎯 Design Elements

### Terminal Window Headers
- Dark background (slate-900)
- Green borders with subtle glow
- Terminal dots (red, yellow, green)
- Animated pulse effects on key elements

### Text Styling
- ALL_CAPS for headers and labels
- `>` prefix for descriptions
- `$` for command prompts
- `#` for IDs and numbers
- Animated cursors (`_`) for active states

### Status Tags
```
READY    - Green background, pulsing
PENDING  - Gray background, static
ERROR    - Red background, pulsing
INFO     - Blue background, static
```

### Interactive Elements
- Green borders on hover
- Glowing shadows (green-500/20)
- Smooth transitions
- Monospace font throughout

## 📦 Component Examples

### Card Structure
```jsx
<div className="bg-slate-900 rounded border-2 border-green-500/20 hover:border-green-500/60 hover:shadow-lg hover:shadow-green-500/20 transition-all p-5">
  {/* Content */}
</div>
```

### Button Styling
```jsx
// Primary Action
<button className="bg-green-500 text-black hover:bg-green-400 font-bold">
  EXECUTE
</button>

// Secondary Action
<button className="bg-slate-700 text-green-400 border border-green-500/30 hover:bg-slate-600 font-bold">
  CANCEL
</button>
```

### Terminal Prompt
```jsx
<div className="flex items-center space-x-2 text-sm font-mono">
  <span className="text-green-500">$</span>
  <span className="text-green-400">command_name</span>
  <span className="animate-pulse">_</span>
</div>
```

### Info Box
```jsx
<div className="bg-slate-900 border border-yellow-500/40 rounded p-4 font-mono">
  <div className="flex items-start space-x-3">
    <Shield className="w-4 h-4 text-yellow-400 animate-pulse" />
    <div>
      <p className="text-yellow-400 font-bold">[!] SYSTEM STATUS</p>
      <p className="text-yellow-300/80">&gt; Status message here</p>
    </div>
  </div>
</div>
```

## 🚀 Animation Guidelines

### Pulse Animations
Used for:
- Active status indicators
- Terminal cursors
- Critical icons (shields, warnings)

```jsx
<Terminal className="animate-pulse" />
```

### Hover Effects
```jsx
hover:border-green-500/60
hover:shadow-lg hover:shadow-green-500/20
hover:bg-green-500/5
```

### Entry Animations (Framer Motion)
```jsx
<motion.div
  initial={{ opacity: 0, x: -20 }}
  animate={{ opacity: 1, x: 0 }}
  transition={{ delay: index * 0.15, type: "spring" }}
>
  {/* Content */}
</motion.div>
```

## 📝 Text Conventions

### Headers
```
PREPROD_TESTING_HUB
FACE_EXTRACTION.EXE
SYSTEM_INFO
```

### Descriptions
```
> Extract and analyze face photos
> Sandboxed testing environment
> Loading models...
```

### Commands
```
$ ls /modules
$ ./extract --source=ID_DOCUMENT
root@preprod:~# command
```

### Status Messages
```
[!] SYSTEM STATUS: ISOLATED ENVIRONMENT
[ERROR] EXTRACTION_FAILED
[INFO] Browser console contains logs
```

### File Paths
```
src/preprod/
http://localhost:3000/preprod
```

## 🎨 Icon Usage

From lucide-react:

- **Terminal**: Main branding, headers
- **Activity**: Active processes
- **Cpu**: Processing/computing
- **Shield**: Security/isolation warnings
- **Zap**: Execute/run actions
- **Scan**: Face/image scanning
- **TestTube2**: Experiments/testing
- **ArrowLeft**: Navigation back

## 🌈 Visual Hierarchy

### Level 1 (Highest)
- Headers: text-green-400, font-bold
- Primary actions: bg-green-500, text-black

### Level 2
- Subheaders: text-cyan-400 or text-green-400
- Secondary actions: bg-slate-700, text-green-400

### Level 3
- Body text: text-green-500/70
- Labels: text-green-500/60

### Level 4 (Lowest)
- Meta info: text-green-500/40
- Disabled: text-slate-700

## 📊 Layout Patterns

### Grid Cards
```jsx
<div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
  {/* Cards */}
</div>
```

### Stats Display
```jsx
<div className="grid grid-cols-2 md:grid-cols-4 gap-3">
  <div className="bg-slate-900 p-3 rounded border border-green-500/30">
    <p className="text-[10px] text-green-500/60">LABEL</p>
    <p className="text-xl font-bold text-green-400">VALUE</p>
  </div>
</div>
```

### Split View
```jsx
<div className="grid grid-cols-1 md:grid-cols-2 divide-y md:divide-y-0 md:divide-x divide-green-500/20">
  <div>{/* Left */}</div>
  <div>{/* Right */}</div>
</div>
```

## 🔧 Utility Classes

### Common Combinations
```jsx
// Container
className="min-h-screen bg-black text-green-400 py-8 px-4"

// Card
className="bg-slate-900 rounded border-2 border-green-500/30 p-6"

// Mono text
className="font-mono text-xs text-green-400"

// Glow effect
className="shadow-lg shadow-green-500/20"
```

## 🎭 Theme Implementation Checklist

When creating new preprod components:

- [ ] Set black background (`bg-black`)
- [ ] Use JetBrains Mono font (`style={{ fontFamily: 'JetBrains Mono, monospace' }}`)
- [ ] Apply green color scheme (green-400, green-500)
- [ ] Use ALL_CAPS for labels
- [ ] Add `>` prefix to descriptions
- [ ] Include terminal-style borders
- [ ] Add subtle glow effects on hover
- [ ] Use monospace-friendly text sizes (text-xs, text-sm)
- [ ] Implement pulse animations for active states
- [ ] Add Framer Motion entry animations

## 🌟 Unique Features

### Terminal Dots
```jsx
<div className="flex space-x-1">
  <div className="w-3 h-3 rounded-full bg-red-500"></div>
  <div className="w-3 h-3 rounded-full bg-yellow-500"></div>
  <div className="w-3 h-3 rounded-full bg-green-500"></div>
</div>
```

### Status Badge
```jsx
<div className="px-2 py-0.5 rounded text-[10px] font-bold bg-green-500 text-black animate-pulse">
  READY
</div>
```

### Cursor Effect
```jsx
<span className="text-green-400 animate-pulse">_</span>
```

## 📱 Responsive Design

- Mobile: Single column, simplified spacing
- Tablet: 2-column grids
- Desktop: 3-column grids, full features

Font sizes scale down on mobile:
- Headers: text-lg → text-base
- Body: text-sm → text-xs
- Labels: text-xs → text-[10px]

---

**Theme Version**: 1.0.0  
**Last Updated**: October 22, 2025  
**Inspired by**: Terminal UI, Matrix aesthetic, Hacker culture


