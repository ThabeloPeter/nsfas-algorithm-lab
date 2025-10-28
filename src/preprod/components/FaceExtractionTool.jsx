'use client';

import { useState, useRef, useEffect } from 'react';
import { Camera, Loader2, AlertCircle, CheckCircle, RefreshCw, Download, CreditCard, BookOpen, User, Shield } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import { extractFaceFromDocument, compareFaces, base64ToBlob } from '@/lib/faceExtractionApi';

export default function FaceExtractionTool() {
  const [step, setStep] = useState('select'); // select, camera, processing, result, selfie-prompt, selfie-camera, comparing, verification-result
  const [idType, setIdType] = useState(null); // 'smart' or 'green'
  const [capturedImage, setCapturedImage] = useState(null);
  const [extractedFaceUrl, setExtractedFaceUrl] = useState(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [error, setError] = useState(null);
  const [extractionData, setExtractionData] = useState(null);
  const [ocrData, setOcrData] = useState(null); // OCR extracted ID fields
  const [stream, setStream] = useState(null);
  const [flashEnabled, setFlashEnabled] = useState(false);
  const [lightingWarning, setLightingWarning] = useState(false);
  const [frameQuality, setFrameQuality] = useState(null); // Real-time frame analysis
  const [feedback, setFeedback] = useState(''); // Progressive feedback message
  const [isAligned, setIsAligned] = useState(false); // Whether ID is properly aligned
  const [isMobile, setIsMobile] = useState(false); // Mobile device detection
  
  // New states for selfie verification
  const [selfieImage, setSelfieImage] = useState(null);
  const [verificationResult, setVerificationResult] = useState(null);
  
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const lightingCheckInterval = useRef(null);
  const frameAnalysisInterval = useRef(null);

  // Detect mobile device
  useEffect(() => {
    const checkMobile = () => {
      setIsMobile(window.innerWidth < 768);
    };
    checkMobile();
    window.addEventListener('resize', checkMobile);
    return () => window.removeEventListener('resize', checkMobile);
  }, []);

  // Start camera
  const startCamera = async () => {
    try {
      const mediaStream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: 'environment', // Use back camera on mobile
          width: { ideal: 1920 },
          height: { ideal: 1080 }
        }
      });
      
      if (videoRef.current) {
        videoRef.current.srcObject = mediaStream;
        setStream(mediaStream);
        
        // Start checking lighting and frame quality after video loads
        videoRef.current.onloadedmetadata = () => {
          startLightingCheck();
          startFrameAnalysis();
        };
      }
    } catch (err) {
      console.error('Camera error:', err);
      setError('Unable to access camera. Please grant camera permissions.');
    }
  };

  // Check lighting conditions
  const checkLighting = () => {
    const video = videoRef.current;
    const canvas = document.createElement('canvas');
    
    if (video && video.readyState === video.HAVE_ENOUGH_DATA) {
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      const ctx = canvas.getContext('2d');
      ctx.drawImage(video, 0, 0);
      
      // Sample center area
      const centerX = canvas.width / 2;
      const centerY = canvas.height / 2;
      const sampleSize = 50;
      
      const imageData = ctx.getImageData(
        centerX - sampleSize / 2,
        centerY - sampleSize / 2,
        sampleSize,
        sampleSize
      );
      
      // Calculate average brightness
      let totalBrightness = 0;
      for (let i = 0; i < imageData.data.length; i += 4) {
        const r = imageData.data[i];
        const g = imageData.data[i + 1];
        const b = imageData.data[i + 2];
        // Calculate perceived brightness
        const brightness = (0.299 * r + 0.587 * g + 0.114 * b);
        totalBrightness += brightness;
      }
      
      const avgBrightness = totalBrightness / (sampleSize * sampleSize);
      
      // Warn if too dark (threshold: 70)
      setLightingWarning(avgBrightness < 70);
    }
  };

  // Start lighting check interval
  const startLightingCheck = () => {
    if (lightingCheckInterval.current) {
      clearInterval(lightingCheckInterval.current);
    }
    lightingCheckInterval.current = setInterval(checkLighting, 1000);
  };

  // Stop lighting check
  const stopLightingCheck = () => {
    if (lightingCheckInterval.current) {
      clearInterval(lightingCheckInterval.current);
      lightingCheckInterval.current = null;
    }
  };

  // Haptic feedback (mobile)
  const triggerHaptic = (type = 'light') => {
    if (navigator.vibrate) {
      if (type === 'success') {
        navigator.vibrate([50, 100, 50]); // Success pattern
      } else if (type === 'error') {
        navigator.vibrate(200); // Error buzz
      } else if (type === 'countdown') {
        navigator.vibrate(30); // Quick tick
      } else {
        navigator.vibrate(50); // Light tap
      }
    }
  };

  // Analyze frame quality in real-time
  const analyzeFrame = () => {
    const video = videoRef.current;
    if (!video || video.readyState !== video.HAVE_ENOUGH_DATA) return;

    const canvas = document.createElement('canvas');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(video, 0, 0);

    const centerX = canvas.width / 2;
    const centerY = canvas.height / 2;
    const sampleSize = 100;

    const imageData = ctx.getImageData(
      centerX - sampleSize / 2,
      centerY - sampleSize / 2,
      sampleSize,
      sampleSize
    );

    // Calculate brightness
    let totalBrightness = 0;
    for (let i = 0; i < imageData.data.length; i += 4) {
      const r = imageData.data[i];
      const g = imageData.data[i + 1];
      const b = imageData.data[i + 2];
      const brightness = (0.299 * r + 0.587 * g + 0.114 * b);
      totalBrightness += brightness;
    }
    const avgBrightness = totalBrightness / (sampleSize * sampleSize);

    // Calculate contrast/edges (simple proxy for alignment)
    const grayData = [];
    for (let i = 0; i < imageData.data.length; i += 4) {
      grayData.push(imageData.data[i]);
    }
    const variance = grayData.reduce((sum, val) => {
      const diff = val - avgBrightness;
      return sum + (diff * diff);
    }, 0) / grayData.length;
    const contrast = Math.sqrt(variance);

    // Determine quality
    const quality = {
      brightness: avgBrightness,
      contrast: contrast,
      isTooDark: avgBrightness < 60,
      isTooBright: avgBrightness > 200,
      isBlurry: contrast < 20,
      isWellLit: avgBrightness >= 80 && avgBrightness <= 180,
      hasGoodContrast: contrast >= 30
    };

    setFrameQuality(quality);

    // Progressive feedback
    if (quality.isTooDark) {
      setFeedback('Too dark - turn on flash or add more light');
      setIsAligned(false);
    } else if (quality.isTooBright) {
      setFeedback('Too bright - move away from direct light');
      setIsAligned(false);
    } else if (quality.isBlurry) {
      setFeedback('Hold steady - image is blurry');
      setIsAligned(false);
    } else if (quality.isWellLit && quality.hasGoodContrast) {
      setFeedback('✓ Perfect! Hold steady...');
      setIsAligned(true);
    } else {
      setFeedback('Position ID in the frame');
      setIsAligned(false);
    }
  };

  // Start frame analysis
  const startFrameAnalysis = () => {
    if (frameAnalysisInterval.current) {
      clearInterval(frameAnalysisInterval.current);
    }
    frameAnalysisInterval.current = setInterval(analyzeFrame, 500); // Check twice per second
  };

  // Stop frame analysis
  const stopFrameAnalysis = () => {
    if (frameAnalysisInterval.current) {
      clearInterval(frameAnalysisInterval.current);
      frameAnalysisInterval.current = null;
    }
    setFrameQuality(null);
    setFeedback('');
    setIsAligned(false);
  };


  // Toggle flash
  const toggleFlash = async () => {
    if (!stream) {
      console.log('❌ No camera stream available');
      return;
    }

    try {
      const track = stream.getVideoTracks()[0];
      const capabilities = track.getCapabilities();
      
      console.log('📸 Camera capabilities:', capabilities);
      console.log('🔦 Torch supported:', !!capabilities.torch);
      
      if (capabilities.torch) {
        try {
          const newFlashState = !flashEnabled;
          console.log(`🔦 Attempting to ${newFlashState ? 'enable' : 'disable'} flash...`);
          
          await track.applyConstraints({
            advanced: [{ torch: newFlashState }]
          });
          
          setFlashEnabled(newFlashState);
          triggerHaptic('light');
          console.log(`✅ Flash ${newFlashState ? 'enabled' : 'disabled'}`);
          
        } catch (err) {
          console.error('❌ Flash constraint error:', err);
          setError(`Flash error: ${err.message || 'Unable to control flash'}`);
        }
      } else {
        console.log('❌ Torch not supported on this device/browser');
        setError('Flash not available. Requirements: Mobile device with back camera on HTTPS.');
      }
    } catch (err) {
      console.error('❌ Flash capability check error:', err);
      setError('Unable to access camera capabilities.');
    }
  };

  // Stop camera
  const stopCamera = () => {
    if (stream) {
      stream.getTracks().forEach(track => track.stop());
      setStream(null);
    }
    stopLightingCheck();
    stopFrameAnalysis();
    setFlashEnabled(false);
    setLightingWarning(false);
  };

  // Cleanup on unmount
  useEffect(() => {
    return () => stopCamera();
  }, []);

  // Select ID type
  const selectIdType = (type) => {
    setIdType(type);
    setStep('camera');
    setTimeout(startCamera, 100);
  };

  // Capture photo
  const capturePhoto = () => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    
    if (video && canvas) {
      triggerHaptic('success'); // Haptic feedback on capture
      
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      const ctx = canvas.getContext('2d');
      ctx.drawImage(video, 0, 0);
      
      canvas.toBlob(async (blob) => {
        const imageUrl = URL.createObjectURL(blob);
        setCapturedImage(imageUrl);
        stopCamera();
        
        // Start extraction
        await extractFace(blob);
      }, 'image/jpeg', 0.95);
    }
  };

  // Extract face
  const extractFace = async (imageBlob) => {
    setStep('processing');
    setIsProcessing(true);
    setError(null);

    try {
      console.log('🚀 Calling face extraction API...');
      console.log('📋 ID Type:', idType);
      
      // Create file from blob
      const file = new File([imageBlob], 'id-photo.jpg', { type: 'image/jpeg' });
      
      // Pass ID type to API for ROI optimization
      const result = await extractFaceFromDocument(file, idType);

      if (!result.success) {
        throw new Error('Face extraction failed');
      }

      setExtractedFaceUrl(result.faceImage);
      setExtractionData({
        totalFaces: result.metadata.total_faces,
        selectedIndex: result.metadata.selected_index,
        confidence: result.metadata.confidence,
        faceSize: result.metadata.face_size,
        positionScore: result.metadata.position_score,
        sizeScore: result.metadata.size_score,
        combinedScore: result.metadata.combined_score,
        allFaces: result.metadata.all_faces
      });

      // Store OCR data
      setOcrData(result.ocrData);

      setStep('result');
      console.log('✅ Face extraction complete!');
      console.log('📝 OCR Data:', result.ocrData);

    } catch (err) {
      console.error('❌ Error:', err);
      triggerHaptic('error'); // Haptic feedback on error
      setError(err.message || 'Face extraction failed. Please try again.');
      setStep('camera');
      setTimeout(startCamera, 100);
    } finally {
      setIsProcessing(false);
    }
  };

  // Retake photo
  const retakePhoto = () => {
    setCapturedImage(null);
    setExtractedFaceUrl(null);
    setExtractionData(null);
    setOcrData(null);
    setError(null);
    setStep('camera');
    setTimeout(startCamera, 100);
  };

  // Reset
  const handleReset = () => {
    stopCamera();
    setCapturedImage(null);
    setExtractedFaceUrl(null);
    setExtractionData(null);
    setOcrData(null);
    setError(null);
    setIdType(null);
    setStep('select');
    if (capturedImage) URL.revokeObjectURL(capturedImage);
    if (extractedFaceUrl) URL.revokeObjectURL(extractedFaceUrl);
    if (selfieImage) URL.revokeObjectURL(selfieImage.url);
    setSelfieImage(null);
    setVerificationResult(null);
  };

  // Start selfie camera
  const startSelfieCamera = () => {
    setStep('selfie-camera');
    setError(null);
    setTimeout(async () => {
      try {
        const mediaStream = await navigator.mediaDevices.getUserMedia({
          video: { 
            facingMode: 'user', // Front camera for selfie
            width: { ideal: 1280 },
            height: { ideal: 720 }
          }
        });
        
        if (videoRef.current) {
          videoRef.current.srcObject = mediaStream;
          setStream(mediaStream);
          startFrameAnalysis(); // Reuse existing frame analysis
        }
      } catch (err) {
        console.error('Camera error:', err);
        setError('Unable to access front camera. Please grant camera permission.');
        setStep('selfie-prompt');
      }
    }, 100);
  };

  // Capture selfie
  const captureSelfie = () => {
    if (!videoRef.current || !canvasRef.current) return;

    const video = videoRef.current;
    const canvas = canvasRef.current;
    
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    
    const ctx = canvas.getContext('2d');
    
    // Flip horizontally to remove mirror effect
    ctx.translate(canvas.width, 0);
    ctx.scale(-1, 1);
    ctx.drawImage(video, 0, 0);
    
    // Reset transform
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    
    canvas.toBlob((blob) => {
      if (!blob) return;
      
      const url = URL.createObjectURL(blob);
      setSelfieImage({ url, blob });
      
      // Stop camera
      stopCamera();
      
      // Trigger haptic feedback
      triggerHaptic('success');
      
      // Start comparison
      compareSelfieWithId(blob);
    }, 'image/jpeg', 0.95);
  };

  // Compare selfie with ID photo
  const compareSelfieWithId = async (selfieBlob) => {
    setStep('comparing');
    setError(null);

    try {
      console.log('🔄 Comparing selfie with ID photo...');
      
      // Convert base64 ID photo to blob
      const idPhotoBlob = base64ToBlob(extractedFaceUrl);
      
      // Create files
      const selfieFile = new File([selfieBlob], 'selfie.jpg', { type: 'image/jpeg' });
      const idPhotoFile = new File([idPhotoBlob], 'id-photo.jpg', { type: 'image/jpeg' });
      
      // Call comparison API
      const result = await compareFaces(selfieFile, idPhotoFile);
      
      if (!result.success) {
        throw new Error('Face comparison failed');
      }
      
      setVerificationResult(result);
      setStep('verification-result');
      
      // Trigger haptic based on result
      triggerHaptic(result.match ? 'success' : 'error');
      
      console.log('✅ Verification complete:', result.match ? 'MATCH' : 'NO MATCH');
      
    } catch (err) {
      console.error('❌ Comparison error:', err);
      setError(err.message || 'Face comparison failed. Please try again.');
      setStep('selfie-prompt');
      triggerHaptic('error');
    }
  };

  // Download
  const handleDownload = () => {
    if (!extractedFaceUrl) return;
    const a = document.createElement('a');
    a.href = extractedFaceUrl;
    a.download = `extracted_face_${Date.now()}.jpg`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  };

  return (
    <div className="min-h-screen bg-white dark:bg-black text-gray-900 dark:text-white transition-colors" style={{ fontFamily: 'JetBrains Mono, monospace' }}>
      <div className="max-w-2xl mx-auto px-4 py-8">
        
        {/* Header */}
        <div className="mb-8 text-center">
          <h1 className="text-2xl md:text-3xl font-bold text-gray-900 dark:text-white tracking-tight mb-2">
            Face Extraction
          </h1>
          <p className="text-gray-600 dark:text-white/60 text-sm font-light">
            Capture your ID to extract the face photo
          </p>
          <div className="h-0.5 w-24 bg-gradient-to-r from-blue-600 dark:from-white to-transparent mt-4 mx-auto"></div>
        </div>

        {/* Error */}
        <AnimatePresence>
          {error && (
            <motion.div
              initial={{ opacity: 0, y: -10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0 }}
              className="mb-6 p-4 bg-red-500/10 border border-red-500/30 rounded-xl"
            >
              <div className="flex items-start space-x-3">
                <AlertCircle className="h-5 w-5 text-red-400 flex-shrink-0 mt-0.5" />
                <div>
                  <p className="text-sm font-semibold text-red-400 mb-1">Error</p>
                  <p className="text-xs text-red-300/90">{error}</p>
                </div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Step 1: Select ID Type */}
        {step === 'select' && (
          <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            className="space-y-4"
          >
            <div className="text-center mb-6">
              <p className="text-gray-700 dark:text-white/80 text-sm mb-2">Select your ID document type:</p>
            </div>

            <button
              onClick={() => selectIdType('smart')}
              className="w-full bg-white dark:bg-white/5 hover:bg-blue-50 dark:hover:bg-white/10 border border-gray-300 dark:border-white/20 hover:border-blue-400 dark:hover:border-white/40 rounded-2xl p-6 transition-all text-left group shadow-sm hover:shadow-md"
            >
              <div className="flex items-start space-x-4">
                <div className="w-16 h-16 rounded-xl bg-blue-100 dark:bg-blue-500/20 flex items-center justify-center flex-shrink-0 group-hover:scale-105 transition-all duration-200">
                  <CreditCard className="w-8 h-8 text-blue-600 dark:text-blue-400" />
                </div>
                <div className="flex-1">
                  <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-1">Smart ID Card</h3>
                  <p className="text-sm text-gray-600 dark:text-white/60 font-light">Card-style ID with photo on the front</p>
                      </div>
                    </div>
            </button>

            <button
              onClick={() => selectIdType('green')}
              className="w-full bg-white dark:bg-white/5 hover:bg-blue-50 dark:hover:bg-white/10 border border-gray-300 dark:border-white/20 hover:border-blue-400 dark:hover:border-white/40 rounded-2xl p-6 transition-all text-left group shadow-sm hover:shadow-md"
            >
              <div className="flex items-start space-x-4">
                <div className="w-16 h-16 rounded-xl bg-green-100 dark:bg-green-500/20 flex items-center justify-center flex-shrink-0 group-hover:scale-105 transition-all duration-200">
                  <BookOpen className="w-8 h-8 text-green-600 dark:text-green-400" />
                </div>
                <div className="flex-1">
                  <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-1">Green ID Book</h3>
                  <p className="text-sm text-gray-600 dark:text-white/60 font-light">Book-style ID with photo on inside page</p>
                  </div>
                </div>
            </button>
          </motion.div>
        )}

        {/* Step 2: Camera View */}
        {step === 'camera' && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className={isMobile ? "fixed inset-0 z-50 bg-white dark:bg-black" : "space-y-4"}
          >
            <div className={isMobile 
              ? "fixed inset-0 w-full h-full" 
              : "relative bg-gray-100 dark:bg-white/5 rounded-2xl overflow-hidden border border-gray-300 dark:border-white/10"
            }>
              <video
                ref={videoRef}
                autoPlay
                playsInline
                muted
                className={isMobile 
                  ? "absolute inset-0 w-full h-full object-cover" 
                  : "w-full h-auto"
                }
              />
              
              {/* Camera Guide Overlay - Different dimensions for each ID type */}
              <div className="absolute inset-0 pointer-events-none">
                <svg className="w-full h-full" viewBox="0 0 100 100" preserveAspectRatio="none">
                  {idType === 'smart' ? (
                    // Smart ID Card - Horizontal card shape (reduced height)
                    <>
                      <rect
                        x="5" y="30" width="90" height="40"
                        fill="none"
                        stroke="rgba(255,255,255,0.5)"
                        strokeWidth="0.3"
                        strokeDasharray="2,2"
                      />
                      {/* Corner markers */}
                      <line x1="5" y1="30" x2="10" y2="30" stroke="white" strokeWidth="0.5" />
                      <line x1="5" y1="30" x2="5" y2="35" stroke="white" strokeWidth="0.5" />
                      
                      <line x1="95" y1="30" x2="90" y2="30" stroke="white" strokeWidth="0.5" />
                      <line x1="95" y1="30" x2="95" y2="35" stroke="white" strokeWidth="0.5" />
                      
                      <line x1="5" y1="70" x2="10" y2="70" stroke="white" strokeWidth="0.5" />
                      <line x1="5" y1="70" x2="5" y2="65" stroke="white" strokeWidth="0.5" />
                      
                      <line x1="95" y1="70" x2="90" y2="70" stroke="white" strokeWidth="0.5" />
                      <line x1="95" y1="70" x2="95" y2="65" stroke="white" strokeWidth="0.5" />
                    </>
                  ) : (
                    // Green ID Book - Square shape (centered)
                    <>
                      <rect
                        x="20" y="30" width="60" height="40"
                        fill="none"
                        stroke="rgba(255,255,255,0.5)"
                        strokeWidth="0.3"
                        strokeDasharray="2,2"
                      />
                      {/* Corner markers */}
                      <line x1="20" y1="30" x2="25" y2="30" stroke="white" strokeWidth="0.5" />
                      <line x1="20" y1="30" x2="20" y2="35" stroke="white" strokeWidth="0.5" />
                      
                      <line x1="80" y1="30" x2="75" y2="30" stroke="white" strokeWidth="0.5" />
                      <line x1="80" y1="30" x2="80" y2="35" stroke="white" strokeWidth="0.5" />
                      
                      <line x1="20" y1="70" x2="25" y2="70" stroke="white" strokeWidth="0.5" />
                      <line x1="20" y1="70" x2="20" y2="65" stroke="white" strokeWidth="0.5" />
                      
                      <line x1="80" y1="70" x2="75" y2="70" stroke="white" strokeWidth="0.5" />
                      <line x1="80" y1="70" x2="80" y2="65" stroke="white" strokeWidth="0.5" />
                    </>
                  )}
                </svg>
              </div>

              {/* Progressive Feedback Overlay */}
              <div className="absolute top-4 left-0 right-0 px-4 space-y-2 z-30">
                <AnimatePresence mode="wait">
                  {feedback && (
                    <motion.div
                      key={feedback}
                      initial={{ opacity: 0, y: -10 }}
                      animate={{ opacity: 1, y: 0 }}
                      exit={{ opacity: 0, y: -10 }}
                      className="text-center"
                    >
                      <div className={`inline-block backdrop-blur-sm px-4 py-2 rounded-full ${
                        isAligned 
                          ? 'bg-green-500/90' 
                          : (feedback.includes('dark') || feedback.includes('bright') || feedback.includes('blurry'))
                            ? 'bg-yellow-500/90'
                            : 'bg-black/70'
                      }`}>
                        <p className={`text-xs font-semibold ${
                          isAligned ? 'text-white' : 
                          (feedback.includes('dark') || feedback.includes('bright') || feedback.includes('blurry'))
                            ? 'text-black'
                            : 'text-white'
                        }`}>
                          {feedback}
                        </p>
                      </div>
                    </motion.div>
                  )}
                </AnimatePresence>
            </div>

              {/* Alignment Indicator */}
              {isAligned && (
                <motion.div
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  className="absolute inset-0 border-4 border-green-500 rounded-2xl pointer-events-none z-20"
                />
              )}

              {/* Close Button (Mobile) */}
              {isMobile && (
                <div className="fixed top-4 left-4 z-50">
                  <button
                    onClick={handleReset}
                    className="p-3 rounded-full bg-black/70 text-white hover:bg-black/80 backdrop-blur-sm transition-all"
                  >
                    <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                    </svg>
                  </button>
            </div>
          )}

              {/* Flash Toggle Button */}
              <div className={isMobile 
                ? "fixed bottom-24 right-4 z-50" 
                : "absolute bottom-4 right-4"
              }>
                <button
                  onClick={toggleFlash}
                  className={`p-3 rounded-full backdrop-blur-sm transition-all ${
                    flashEnabled 
                      ? 'bg-yellow-500 text-black' 
                      : 'bg-black/70 text-white hover:bg-black/80'
                  }`}
                >
                  <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 24 24">
                    <path d="M7 2v11h3v9l7-12h-4l4-8z"/>
                  </svg>
                </button>
              </div>
            </div>

            {/* Capture Button */}
            <div className={isMobile
              ? "fixed bottom-4 left-4 right-4 z-50 flex space-x-3"
              : "flex space-x-3"
            }>
              <button
                onClick={capturePhoto}
                disabled={!stream}
                className="flex-1 bg-white text-black py-4 rounded-xl font-semibold hover:bg-white/90 transition-all flex items-center justify-center space-x-2 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                <Camera className="w-5 h-5" />
                <span>Capture Photo</span>
              </button>
              {!isMobile && (
                <button
                  onClick={handleReset}
                  className="px-6 bg-white/10 text-white rounded-xl border border-white/20 hover:bg-white/20 transition-all"
                >
                  Cancel
                </button>
              )}
            </div>

            {/* Tips */}
            {!isMobile && (
              <div className="bg-blue-500/10 border border-blue-500/20 rounded-xl p-4">
                <p className="text-xs text-blue-300 font-semibold mb-2">Tips for best results:</p>
                <ul className="text-xs text-blue-200/80 space-y-1 font-light">
                  <li>• <strong>Real-time Guidance:</strong> Follow on-screen instructions</li>
                  <li>• <strong>Quality Check:</strong> We analyze lighting & focus</li>
                  <li>• <strong>Alignment:</strong> Green border shows when aligned</li>
                  <li>• <strong>Ready to Capture:</strong> Tap "Capture Photo" when ready</li>
                </ul>
              </div>
            )}
          </motion.div>
        )}

        {/* Step 3: Processing */}
        {step === 'processing' && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="bg-white dark:bg-white/5 rounded-2xl border border-gray-300 dark:border-white/10 p-12 text-center shadow-lg"
          >
            <Loader2 className="w-16 h-16 text-blue-600 dark:text-white animate-spin mx-auto mb-6" />
            <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-2">Processing</h3>
            <p className="text-sm text-gray-600 dark:text-white/60 font-light">
              Extracting face from your ID document...
            </p>
          </motion.div>
        )}

        {/* Step 4: Results */}
        {step === 'result' && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="space-y-4"
          >
            {/* Extracted Face */}
            <div className="bg-white dark:bg-white/5 rounded-2xl border border-gray-300 dark:border-white/10 p-6 shadow-lg">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-sm font-semibold text-gray-900 dark:text-white">Extracted Face</h3>
                <CheckCircle className="w-5 h-5 text-green-500 dark:text-green-400" />
                </div>

              <div className="relative aspect-square max-w-xs mx-auto bg-gray-100 dark:bg-white/5 rounded-xl border border-gray-300 dark:border-white/20 overflow-hidden mb-4">
                    <img
                      src={extractedFaceUrl}
                      alt="Extracted face"
                      className="w-full h-full object-cover"
                    />
                    </div>

                    <button
                      onClick={handleDownload}
                className="w-full py-3 bg-blue-600 hover:bg-blue-700 dark:bg-white text-white dark:text-black rounded-lg dark:hover:bg-white/90 transition-all font-medium text-sm flex items-center justify-center space-x-2"
                    >
                      <Download className="w-4 h-4" />
                <span>Download Face</span>
                    </button>
              </div>

              {/* Stats */}
              {extractionData && (
              <div className="bg-white dark:bg-white/5 rounded-2xl border border-gray-300 dark:border-white/10 p-6 shadow-lg">
                <h4 className="text-sm font-semibold text-gray-900 dark:text-white mb-4">Extraction Details</h4>
                
                <div className="grid grid-cols-2 gap-3 mb-4">
                  <div className="bg-gray-100 dark:bg-white/5 p-3 rounded-lg">
                    <p className="text-xs text-gray-500 dark:text-white/50 mb-1">Faces Detected</p>
                    <p className="text-xl font-semibold text-gray-900 dark:text-white">{extractionData.totalFaces}</p>
                    </div>
                  <div className="bg-gray-100 dark:bg-white/5 p-3 rounded-lg">
                    <p className="text-xs text-gray-500 dark:text-white/50 mb-1">Confidence</p>
                    <p className="text-xl font-semibold text-gray-900 dark:text-white">{extractionData.confidence}%</p>
                    </div>
                  </div>

                <div className="bg-gray-100 dark:bg-white/5 p-3 rounded-lg text-xs">
                  <p className="text-gray-500 dark:text-white/50 mb-2">Selection Score</p>
                  <div className="space-y-1">
                    <div className="flex justify-between">
                      <span className="text-gray-600 dark:text-white/60">Size Score:</span>
                      <span className="text-gray-900 dark:text-white font-semibold">{extractionData.sizeScore}%</span>
                      </div>
                    <div className="flex justify-between">
                      <span className="text-gray-600 dark:text-white/60">Position Score:</span>
                      <span className="text-gray-900 dark:text-white font-semibold">{extractionData.positionScore}%</span>
                      </div>
                    </div>
                  </div>
                </div>
              )}

              {/* OCR Extracted ID Details */}
              {ocrData && ocrData.success && (
                <div className="bg-white dark:bg-white/5 rounded-2xl border border-gray-300 dark:border-white/10 p-6 shadow-lg">
                  <div className="flex items-center justify-between mb-4">
                    <h4 className="text-sm font-semibold text-gray-900 dark:text-white">ID Details (OCR)</h4>
                    <span className="text-xs bg-green-500/20 text-green-400 px-2 py-1 rounded-full">
                      {ocrData.fieldsExtracted}/{ocrData.totalFields} Fields
                    </span>
                  </div>

                  <div className="space-y-3">
                    {Object.entries(ocrData.fields).map(([key, field]) => {
                      const displayName = key === 'idNumber' ? 'ID Number' :
                                        key === 'countryOfBirth' ? 'Country of Birth' :
                                        key === 'dateIssued' ? 'Date Issued' :
                                        key === 'placeOfBirth' ? 'Place of Birth' :
                                        key.charAt(0).toUpperCase() + key.slice(1);
                      
                      const statusIcon = field.status === 'success' ? '✓' :
                                        field.status === 'partial' ? '⚠' : '✗';
                      
                      const statusColor = field.status === 'success' ? 'text-green-400' :
                                         field.status === 'partial' ? 'text-yellow-400' :
                                         'text-gray-500 dark:text-white/30';

                      return (
                        <div key={key} className="bg-gray-50 dark:bg-white/5 p-3 rounded-lg">
                          <div className="flex items-center justify-between mb-1">
                            <p className="text-xs text-gray-500 dark:text-white/50">{displayName}</p>
                            <span className={`text-sm font-semibold ${statusColor}`}>{statusIcon}</span>
                          </div>
                          <p className="text-sm font-semibold text-gray-900 dark:text-white">
                            {field.value || (
                              <span className="text-gray-400 dark:text-white/40 italic">
                                {field.status === 'not_detected' ? 'Not detected' : 'Unable to read'}
                              </span>
                            )}
                          </p>
                          {field.confidence > 0 && (
                            <p className="text-xs text-gray-400 dark:text-white/40 mt-1">
                              Confidence: {field.confidence}%
                            </p>
                          )}
                        </div>
                      );
                    })}
                  </div>

                  {/* Overall OCR Confidence */}
                  <div className="mt-4 pt-4 border-t border-gray-200 dark:border-white/10">
                    <div className="flex items-center justify-between">
                      <span className="text-xs text-gray-500 dark:text-white/50">Overall OCR Confidence</span>
                      <span className="text-sm font-semibold text-gray-900 dark:text-white">{ocrData.confidence}%</span>
                    </div>
                  </div>
                </div>
              )}

              {/* OCR Failed/Partial Message */}
              {ocrData && !ocrData.success && (
                <div className="bg-yellow-500/10 border border-yellow-500/20 rounded-xl p-4">
                  <div className="flex items-start space-x-3">
                    <AlertCircle className="w-5 h-5 text-yellow-400 flex-shrink-0 mt-0.5" />
                    <div>
                      <p className="text-sm font-semibold text-white mb-1">OCR Extraction Limited</p>
                      <p className="text-xs text-white/60 font-light">
                        {ocrData.message || 'Unable to extract all ID details. The face was successfully extracted.'}
                      </p>
                    </div>
                  </div>
                </div>
              )}

              {/* Actions */}
              <div className="space-y-3">
                {/* Verification Prompt */}
                <div className="bg-blue-500/10 border border-blue-500/20 rounded-xl p-4">
                  <div className="flex items-start space-x-3 mb-3">
                    <Shield className="w-5 h-5 text-blue-400 flex-shrink-0 mt-0.5" />
                    <div className="flex-1">
                      <p className="text-sm font-semibold text-white mb-1">
                        Next Step: Verify Identity
                      </p>
                      <p className="text-xs text-white/60 font-light">
                        Take a selfie to confirm this is really you
                      </p>
                    </div>
                  </div>
                  <button
                    onClick={() => setStep('selfie-prompt')}
                    className="w-full py-3 bg-blue-600 hover:bg-blue-700 text-white rounded-lg font-medium text-sm flex items-center justify-center space-x-2 transition-all"
                  >
                    <User className="w-4 h-4" />
                    <span>Continue to Verification</span>
                  </button>
                </div>

                <button
                  onClick={handleReset}
                  className="w-full py-4 bg-gray-100 dark:bg-white/10 text-gray-900 dark:text-white rounded-xl border border-gray-300 dark:border-white/20 hover:bg-gray-200 dark:hover:bg-white/20 font-medium flex items-center justify-center space-x-2 transition-all"
                >
                  <RefreshCw className="w-4 h-4" />
                  <span>Start Over</span>
                </button>
              </div>
          </motion.div>
          )}

        {/* Step 5: Selfie Prompt */}
        {step === 'selfie-prompt' && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="space-y-4"
          >
            <div className="bg-white dark:bg-white/5 rounded-2xl border border-gray-300 dark:border-white/10 p-6 shadow-lg">
              <div className="flex items-center justify-center mb-6">
                <div className="w-16 h-16 rounded-full bg-blue-100 dark:bg-blue-500/20 flex items-center justify-center">
                  <Shield className="w-8 h-8 text-blue-600 dark:text-blue-400" />
                </div>
              </div>

              <h3 className="text-xl font-semibold text-gray-900 dark:text-white text-center mb-2">
                Identity Verification
              </h3>
              <p className="text-sm text-gray-600 dark:text-white/60 text-center mb-6 font-light">
                Let's verify this is really you
              </p>

              {/* ID Photo Preview */}
              <div className="mb-6">
                <p className="text-xs text-gray-500 dark:text-white/50 text-center mb-2">Your ID Photo</p>
                <div className="relative w-32 h-32 mx-auto bg-gray-100 dark:bg-white/5 rounded-xl border border-gray-300 dark:border-white/20 overflow-hidden">
                  <img
                    src={extractedFaceUrl}
                    alt="ID face"
                    className="w-full h-full object-cover"
                  />
                </div>
              </div>

              <div className="bg-blue-50 dark:bg-blue-500/10 border border-blue-200 dark:border-blue-500/20 rounded-xl p-4 mb-6">
                <p className="text-sm text-gray-900 dark:text-white/90 mb-3 font-medium">Ready to verify?</p>
                <p className="text-xs text-gray-600 dark:text-white/60 mb-3 font-light">
                  We'll compare your live selfie with the ID photo above
                </p>
                <div className="space-y-1.5 text-xs text-gray-600 dark:text-white/60">
                  <div className="flex items-center space-x-2">
                    <div className="w-1.5 h-1.5 rounded-full bg-blue-500 dark:bg-blue-400"></div>
                    <span>Face the camera directly</span>
                  </div>
                  <div className="flex items-center space-x-2">
                    <div className="w-1.5 h-1.5 rounded-full bg-blue-500 dark:bg-blue-400"></div>
                    <span>Ensure good lighting</span>
                  </div>
                  <div className="flex items-center space-x-2">
                    <div className="w-1.5 h-1.5 rounded-full bg-blue-500 dark:bg-blue-400"></div>
                    <span>Remove glasses if possible</span>
                  </div>
                  <div className="flex items-center space-x-2">
                    <div className="w-1.5 h-1.5 rounded-full bg-blue-500 dark:bg-blue-400"></div>
                    <span>Use a neutral expression</span>
                  </div>
                </div>
              </div>

              <div className="space-y-3">
                <button
                  onClick={startSelfieCamera}
                  className="w-full py-4 bg-blue-600 hover:bg-blue-700 text-white rounded-xl font-semibold flex items-center justify-center space-x-2 transition-all"
                >
                  <Camera className="w-5 h-5" />
                  <span>Take Selfie</span>
                </button>
                <button
                  onClick={() => setStep('result')}
                  className="w-full py-3 bg-gray-100 dark:bg-white/10 text-gray-900 dark:text-white rounded-xl border border-gray-300 dark:border-white/20 hover:bg-gray-200 dark:hover:bg-white/20 font-medium flex items-center justify-center space-x-2 transition-all"
                >
                  <span>Back to Results</span>
                </button>
              </div>
            </div>
          </motion.div>
        )}

        {/* Step 6: Selfie Camera */}
        {step === 'selfie-camera' && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className={isMobile ? "fixed inset-0 z-50 bg-white dark:bg-black" : "space-y-4"}
          >
            <div className={isMobile 
              ? "fixed inset-0 w-full h-full" 
              : "relative bg-gray-100 dark:bg-white/5 rounded-2xl overflow-hidden border border-gray-300 dark:border-white/10"
            }>
              <video
                ref={videoRef}
                autoPlay
                playsInline
                muted
                className={isMobile 
                  ? "absolute inset-0 w-full h-full object-cover" 
                  : "w-full h-auto"
                }
                style={{ transform: 'scaleX(-1)' }}
              />
              
              {/* Oval Face Guide for Selfie */}
              <div className="absolute inset-0 pointer-events-none" style={{ transform: 'scaleX(-1)' }}>
                <svg className="w-full h-full" viewBox="0 0 100 100" preserveAspectRatio="none">
                  <ellipse 
                    cx="50" 
                    cy="45" 
                    rx="30" 
                    ry="38"
                    fill="none"
                    stroke="rgba(255,255,255,0.5)"
                    strokeWidth="0.3"
                    strokeDasharray="2,2"
                  />
                  {/* Guide text */}
                  <text x="50" y="90" fontSize="3" fill="white" textAnchor="middle" opacity="0.7">
                    Center your face in the oval
                  </text>
                </svg>
              </div>

              {/* Progressive Feedback */}
              <div className="absolute top-4 left-0 right-0 px-4 space-y-2 z-30">
                <AnimatePresence>
                  {feedback && (
                    <motion.div
                      initial={{ opacity: 0, y: -10 }}
                      animate={{ opacity: 1, y: 0 }}
                      exit={{ opacity: 0 }}
                      className="bg-black/70 backdrop-blur-sm text-white px-4 py-2 rounded-lg text-sm text-center border border-white/20"
                    >
                      {feedback}
                    </motion.div>
                  )}
                </AnimatePresence>
              </div>

              {/* Alignment Indicator */}
              {isAligned && (
                <motion.div
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  className="absolute inset-0 border-4 border-green-500 rounded-2xl pointer-events-none z-20"
                />
              )}

              {/* Close Button (Mobile) */}
              {isMobile && (
                <div className="fixed top-4 left-4 z-50">
                  <button
                    onClick={() => { stopCamera(); setStep('selfie-prompt'); }}
                    className="p-3 rounded-full bg-black/70 text-white hover:bg-black/80 backdrop-blur-sm transition-all"
                  >
                    <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                    </svg>
                  </button>
                </div>
              )}

              {/* Flash Toggle Button */}
              <div className={isMobile 
                ? "fixed bottom-24 right-4 z-50" 
                : "absolute bottom-4 right-4"
              }>
                <button
                  onClick={toggleFlash}
                  className={`p-3 rounded-full backdrop-blur-sm transition-all ${
                    flashEnabled 
                      ? 'bg-yellow-500 text-black' 
                      : 'bg-black/70 text-white hover:bg-black/80'
                  }`}
                >
                  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                  </svg>
                </button>
              </div>
            </div>

            {/* Capture Button */}
            <div className={isMobile
              ? "fixed bottom-4 left-4 right-4 z-50 flex space-x-3"
              : "flex space-x-3"
            }>
              <button
                onClick={captureSelfie}
                disabled={!stream}
                className="flex-1 bg-white text-black py-4 rounded-xl font-semibold hover:bg-white/90 transition-all flex items-center justify-center space-x-2 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                <Camera className="w-5 h-5" />
                <span>Capture Selfie</span>
              </button>
              {!isMobile && (
                <button
                  onClick={() => { stopCamera(); setStep('selfie-prompt'); }}
                  className="px-6 bg-white/10 text-white rounded-xl border border-white/20 hover:bg-white/20 transition-all"
                >
                  Cancel
                </button>
              )}
            </div>
          </motion.div>
        )}

        {/* Step 7: Comparing */}
        {step === 'comparing' && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="bg-white dark:bg-white/5 rounded-2xl border border-gray-300 dark:border-white/10 p-12 text-center shadow-lg"
          >
            <div className="flex items-center justify-center space-x-8 mb-8">
              {/* ID Photo */}
              <div className="text-center">
                <div className="w-24 h-24 rounded-xl border-2 border-gray-300 dark:border-white/20 overflow-hidden mb-2">
                  <img src={extractedFaceUrl} alt="ID" className="w-full h-full object-cover" />
                </div>
                <p className="text-xs text-gray-500 dark:text-white/50">ID Photo</p>
              </div>

              {/* Comparison Icon */}
              <div className="relative">
                <motion.div
                  animate={{ rotate: 360 }}
                  transition={{ duration: 2, repeat: Infinity, ease: "linear" }}
                >
                  <Loader2 className="w-12 h-12 text-blue-600 dark:text-blue-400" />
                </motion.div>
              </div>

              {/* Selfie */}
              <div className="text-center">
                <div className="w-24 h-24 rounded-xl border-2 border-gray-300 dark:border-white/20 overflow-hidden mb-2">
                  <img src={selfieImage?.url} alt="Selfie" className="w-full h-full object-cover" />
                </div>
                <p className="text-xs text-gray-500 dark:text-white/50">Your Selfie</p>
              </div>
            </div>

            <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-2">Verifying Identity...</h3>
            <p className="text-sm text-gray-600 dark:text-white/60 font-light">
              Comparing faces using AI (99.38% accuracy)
            </p>
          </motion.div>
        )}

        {/* Step 8: Verification Result */}
        {step === 'verification-result' && verificationResult && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="space-y-4"
          >
            {/* Result Header */}
            <div className={`rounded-2xl border p-6 text-center ${
              verificationResult.match 
                ? 'bg-green-500/10 border-green-500/20' 
                : 'bg-red-500/10 border-red-500/20'
            }`}>
              <div className="flex items-center justify-center mb-4">
                <div className={`w-16 h-16 rounded-full flex items-center justify-center ${
                  verificationResult.match ? 'bg-green-500/20' : 'bg-red-500/20'
                }`}>
                  {verificationResult.match ? (
                    <CheckCircle className="w-8 h-8 text-green-400" />
                  ) : (
                    <AlertCircle className="w-8 h-8 text-red-400" />
          )}
        </div>
              </div>

              <h3 className={`text-2xl font-bold mb-2 ${
                verificationResult.match ? 'text-green-400' : 'text-red-400'
              }`}>
                {verificationResult.match ? 'Identity Verified!' : 'Verification Failed'}
              </h3>
              <p className="text-sm text-white/60 font-light">
                {verificationResult.match 
                  ? 'The photos match! Identity confirmed.' 
                  : 'The photos don\'t match or quality is too low.'}
              </p>
            </div>

            {/* Comparison Visual */}
            <div className="bg-white dark:bg-white/5 rounded-2xl border border-gray-300 dark:border-white/10 p-6 shadow-lg">
              <div className="flex items-center justify-center space-x-6 mb-6">
                <div className="text-center">
                  <div className="w-28 h-28 rounded-xl border-2 border-gray-300 dark:border-white/20 overflow-hidden mb-2">
                    <img src={extractedFaceUrl} alt="ID" className="w-full h-full object-cover" />
                  </div>
                  <p className="text-xs text-gray-500 dark:text-white/50">ID Photo</p>
                </div>

                <div className="text-2xl text-gray-900 dark:text-white">
                  {verificationResult.match ? '✓' : '✗'}
                </div>

                <div className="text-center">
                  <div className="w-28 h-28 rounded-xl border-2 border-gray-300 dark:border-white/20 overflow-hidden mb-2">
                    <img src={selfieImage?.url} alt="Selfie" className="w-full h-full object-cover" />
                  </div>
                  <p className="text-xs text-gray-500 dark:text-white/50">Your Selfie</p>
                </div>
              </div>

              {/* Metrics */}
              <div className="grid grid-cols-3 gap-3">
                <div className="bg-gray-100 dark:bg-white/5 p-3 rounded-lg text-center">
                  <p className="text-xs text-gray-500 dark:text-white/50 mb-1">Similarity</p>
                  <p className="text-lg font-semibold text-gray-900 dark:text-white">{verificationResult.similarity}%</p>
                </div>
                <div className="bg-gray-100 dark:bg-white/5 p-3 rounded-lg text-center">
                  <p className="text-xs text-gray-500 dark:text-white/50 mb-1">Confidence</p>
                  <p className="text-lg font-semibold text-gray-900 dark:text-white">{verificationResult.confidence}%</p>
                </div>
                <div className="bg-gray-100 dark:bg-white/5 p-3 rounded-lg text-center">
                  <p className="text-xs text-gray-500 dark:text-white/50 mb-1">Distance</p>
                  <p className="text-lg font-semibold text-gray-900 dark:text-white">{verificationResult.distance}</p>
                </div>
              </div>
            </div>

            {/* Technical Details */}
            <div className="bg-gray-100 dark:bg-white/5 rounded-xl border border-gray-300 dark:border-white/10 p-4 text-xs text-gray-600 dark:text-white/60">
              <p className="mb-1"><strong className="text-gray-900 dark:text-white">Method:</strong> face_recognition (dlib ResNet)</p>
              <p className="mb-1"><strong className="text-gray-900 dark:text-white">Accuracy:</strong> 99.38% (LFW benchmark)</p>
              <p><strong className="text-gray-900 dark:text-white">Threshold:</strong> {verificationResult.threshold}</p>
            </div>

            {/* Actions */}
            <div className="space-y-3">
              {!verificationResult.match && (
                <button
                  onClick={startSelfieCamera}
                  className="w-full py-4 bg-blue-600 hover:bg-blue-700 text-white rounded-xl font-semibold flex items-center justify-center space-x-2 transition-all"
                >
                  <RefreshCw className="w-4 h-4" />
                  <span>Retry Selfie</span>
                </button>
              )}
              <button
                onClick={handleReset}
                className="w-full py-4 bg-gray-100 dark:bg-white/10 text-gray-900 dark:text-white rounded-xl border border-gray-300 dark:border-white/20 hover:bg-gray-200 dark:hover:bg-white/20 font-medium flex items-center justify-center space-x-2 transition-all"
              >
                <RefreshCw className="w-4 h-4" />
                <span>Verify Another Person</span>
              </button>
            </div>
          </motion.div>
        )}

        {/* Hidden canvas for capture */}
        <canvas ref={canvasRef} className="hidden" />
      </div>
    </div>
  );
}
