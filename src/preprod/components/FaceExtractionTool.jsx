'use client';

import { useState, useRef, useEffect } from 'react';
import { Camera, Loader2, AlertCircle, CheckCircle, RefreshCw, Download, CreditCard, BookOpen } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import { extractFaceFromDocument } from '@/lib/faceExtractionApi';

export default function FaceExtractionTool() {
  const [step, setStep] = useState('select'); // select, camera, processing, result
  const [idType, setIdType] = useState(null); // 'smart' or 'green'
  const [capturedImage, setCapturedImage] = useState(null);
  const [extractedFaceUrl, setExtractedFaceUrl] = useState(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [error, setError] = useState(null);
  const [extractionData, setExtractionData] = useState(null);
  const [stream, setStream] = useState(null);
  const [flashEnabled, setFlashEnabled] = useState(false);
  const [lightingWarning, setLightingWarning] = useState(false);
  const [frameQuality, setFrameQuality] = useState(null); // Real-time frame analysis
  const [feedback, setFeedback] = useState(''); // Progressive feedback message
  const [countdown, setCountdown] = useState(null); // Auto-capture countdown
  const [isAligned, setIsAligned] = useState(false); // Whether ID is properly aligned
  const [isMobile, setIsMobile] = useState(false); // Mobile device detection
  
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const lightingCheckInterval = useRef(null);
  const frameAnalysisInterval = useRef(null);
  const countdownTimer = useRef(null);

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
      setFeedback('💡 Too dark - turn on flash or add more light');
      setIsAligned(false);
    } else if (quality.isTooBright) {
      setFeedback('☀️ Too bright - move away from direct light');
      setIsAligned(false);
    } else if (quality.isBlurry) {
      setFeedback('📷 Hold steady - image is blurry');
      setIsAligned(false);
    } else if (quality.isWellLit && quality.hasGoodContrast) {
      setFeedback('✓ Perfect! Hold steady...');
      setIsAligned(true);
    } else {
      setFeedback('📸 Position ID in the frame');
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

  // Start auto-capture countdown
  const startCountdown = () => {
    if (countdownTimer.current) return; // Already counting

    setCountdown(3);
    triggerHaptic('countdown');

    countdownTimer.current = setInterval(() => {
      setCountdown(prev => {
        if (prev === null) return null;
        
        if (prev <= 1) {
          clearInterval(countdownTimer.current);
          countdownTimer.current = null;
          capturePhoto();
          return null;
        }
        
        triggerHaptic('countdown');
        return prev - 1;
      });
    }, 1000);
  };

  // Cancel countdown
  const cancelCountdown = () => {
    if (countdownTimer.current) {
      clearInterval(countdownTimer.current);
      countdownTimer.current = null;
    }
    setCountdown(null);
  };

  // Auto-capture when aligned
  useEffect(() => {
    if (step === 'camera' && isAligned && !countdown) {
      // Start countdown when conditions are perfect
      const timer = setTimeout(() => {
        if (isAligned) {
          startCountdown();
        }
      }, 1000); // Wait 1 second of stability

      return () => clearTimeout(timer);
    } else if (!isAligned && countdown) {
      // Cancel countdown if alignment lost
      cancelCountdown();
    }
  }, [isAligned, countdown, step]);

  // Toggle flash
  const toggleFlash = async () => {
    if (stream) {
      const track = stream.getVideoTracks()[0];
      const capabilities = track.getCapabilities();
      
      if (capabilities.torch) {
        try {
          await track.applyConstraints({
            advanced: [{ torch: !flashEnabled }]
          });
          setFlashEnabled(!flashEnabled);
        } catch (err) {
          console.error('Flash error:', err);
          setError('Unable to control flash on this device.');
        }
      } else {
        setError('Flash not available on this device.');
      }
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
    cancelCountdown();
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

      setStep('result');
      console.log('✅ Face extraction complete!');

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
    setError(null);
    setIdType(null);
    setStep('select');
    if (capturedImage) URL.revokeObjectURL(capturedImage);
    if (extractedFaceUrl) URL.revokeObjectURL(extractedFaceUrl);
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
    <div className="min-h-screen bg-black text-white" style={{ fontFamily: 'JetBrains Mono, monospace' }}>
      <div className="max-w-2xl mx-auto px-4 py-8">
        
        {/* Header */}
        <div className="mb-8 text-center">
          <h1 className="text-2xl md:text-3xl font-bold text-white tracking-tight mb-2">
            Face Extraction
          </h1>
          <p className="text-white/60 text-sm font-light">
            Capture your ID to extract the face photo
          </p>
          <div className="h-0.5 w-24 bg-gradient-to-r from-white to-transparent mt-4 mx-auto"></div>
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
              <p className="text-white/80 text-sm mb-2">Select your ID document type:</p>
            </div>

            <button
              onClick={() => selectIdType('smart')}
              className="w-full bg-white/5 hover:bg-white/10 border border-white/20 hover:border-white/40 rounded-2xl p-6 transition-all text-left group"
            >
              <div className="flex items-start space-x-4">
                <div className="w-12 h-12 bg-white/10 rounded-lg flex items-center justify-center flex-shrink-0 group-hover:bg-white/20 transition-colors">
                  <CreditCard className="w-6 h-6 text-white" />
                </div>
                <div className="flex-1">
                  <h3 className="text-lg font-semibold text-white mb-1">Smart ID Card</h3>
                  <p className="text-sm text-white/60 font-light">Card-style ID with photo on the front</p>
                      </div>
                    </div>
            </button>

            <button
              onClick={() => selectIdType('green')}
              className="w-full bg-white/5 hover:bg-white/10 border border-white/20 hover:border-white/40 rounded-2xl p-6 transition-all text-left group"
            >
              <div className="flex items-start space-x-4">
                <div className="w-12 h-12 bg-white/10 rounded-lg flex items-center justify-center flex-shrink-0 group-hover:bg-white/20 transition-colors">
                  <BookOpen className="w-6 h-6 text-white" />
                </div>
                <div className="flex-1">
                  <h3 className="text-lg font-semibold text-white mb-1">Green ID Book</h3>
                  <p className="text-sm text-white/60 font-light">Book-style ID with photo on inside page</p>
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
            className={isMobile ? "fixed inset-0 z-50 bg-black" : "space-y-4"}
          >
            <div className={isMobile 
              ? "fixed inset-0 w-full h-full" 
              : "relative bg-white/5 rounded-2xl overflow-hidden border border-white/10"
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
                    // Smart ID Card - Horizontal card shape (85.6mm x 54mm ratio)
                    <>
                      <rect
                        x="5" y="25" width="90" height="50"
                        fill="none"
                        stroke="rgba(255,255,255,0.5)"
                        strokeWidth="0.3"
                        strokeDasharray="2,2"
                      />
                      {/* Corner markers */}
                      <line x1="5" y1="25" x2="10" y2="25" stroke="white" strokeWidth="0.5" />
                      <line x1="5" y1="25" x2="5" y2="30" stroke="white" strokeWidth="0.5" />
                      
                      <line x1="95" y1="25" x2="90" y2="25" stroke="white" strokeWidth="0.5" />
                      <line x1="95" y1="25" x2="95" y2="30" stroke="white" strokeWidth="0.5" />
                      
                      <line x1="5" y1="75" x2="10" y2="75" stroke="white" strokeWidth="0.5" />
                      <line x1="5" y1="75" x2="5" y2="70" stroke="white" strokeWidth="0.5" />
                      
                      <line x1="95" y1="75" x2="90" y2="75" stroke="white" strokeWidth="0.5" />
                      <line x1="95" y1="75" x2="95" y2="70" stroke="white" strokeWidth="0.5" />
                    </>
                  ) : (
                    // Green ID Book - Square shape
                    <>
                      <rect
                        x="10" y="10" width="80" height="80"
                        fill="none"
                        stroke="rgba(255,255,255,0.5)"
                        strokeWidth="0.3"
                        strokeDasharray="2,2"
                      />
                      {/* Corner markers */}
                      <line x1="10" y1="10" x2="15" y2="10" stroke="white" strokeWidth="0.5" />
                      <line x1="10" y1="10" x2="10" y2="15" stroke="white" strokeWidth="0.5" />
                      
                      <line x1="90" y1="10" x2="85" y2="10" stroke="white" strokeWidth="0.5" />
                      <line x1="90" y1="10" x2="90" y2="15" stroke="white" strokeWidth="0.5" />
                      
                      <line x1="10" y1="90" x2="15" y2="90" stroke="white" strokeWidth="0.5" />
                      <line x1="10" y1="90" x2="10" y2="85" stroke="white" strokeWidth="0.5" />
                      
                      <line x1="90" y1="90" x2="85" y2="90" stroke="white" strokeWidth="0.5" />
                      <line x1="90" y1="90" x2="90" y2="85" stroke="white" strokeWidth="0.5" />
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

              {/* Countdown Overlay */}
              {countdown !== null && (
                <motion.div
                  initial={{ scale: 0.5, opacity: 0 }}
                  animate={{ scale: 1, opacity: 1 }}
                  exit={{ scale: 1.5, opacity: 0 }}
                  className="absolute inset-0 flex items-center justify-center bg-black/50 backdrop-blur-sm z-40"
                >
                  <motion.div
                    key={countdown}
                    initial={{ scale: 0.8, opacity: 0 }}
                    animate={{ scale: 1, opacity: 1 }}
                    exit={{ scale: 1.2, opacity: 0 }}
                    className="text-9xl font-bold text-white drop-shadow-2xl"
                  >
                    {countdown}
                  </motion.div>
                </motion.div>
              )}

              {/* Alignment Indicator */}
              {isAligned && !countdown && (
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
              {countdown !== null ? (
                <button
                  onClick={cancelCountdown}
                  className="flex-1 bg-red-500 text-white py-4 rounded-xl font-semibold hover:bg-red-600 transition-all flex items-center justify-center space-x-2"
                >
                  <AlertCircle className="w-5 h-5" />
                  <span>Cancel Auto-Capture</span>
                </button>
              ) : (
                <button
                  onClick={capturePhoto}
                  disabled={!stream}
                  className="flex-1 bg-white text-black py-4 rounded-xl font-semibold hover:bg-white/90 transition-all flex items-center justify-center space-x-2 disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  <Camera className="w-5 h-5" />
                  <span>Capture Now</span>
                </button>
              )}
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
                <p className="text-xs text-blue-300 font-semibold mb-2">✨ Smart Capture Features:</p>
                <ul className="text-xs text-blue-200/80 space-y-1 font-light">
                  <li>• <strong>Auto-Capture:</strong> Aligns perfectly? We'll capture automatically!</li>
                  <li>• <strong>Real-time Guidance:</strong> Follow on-screen instructions</li>
                  <li>• <strong>Quality Check:</strong> We ensure optimal lighting & focus</li>
                  <li>• <strong>Manual Override:</strong> Tap "Capture Now" anytime</li>
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
            className="bg-white/5 rounded-2xl border border-white/10 p-12 text-center"
          >
            <Loader2 className="w-16 h-16 text-white animate-spin mx-auto mb-6" />
            <h3 className="text-lg font-semibold text-white mb-2">Processing</h3>
            <p className="text-sm text-white/60 font-light">
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
            <div className="bg-white/5 rounded-2xl border border-white/10 p-6">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-sm font-semibold text-white">Extracted Face</h3>
                <CheckCircle className="w-5 h-5 text-green-400" />
                </div>

              <div className="relative aspect-square max-w-xs mx-auto bg-white/5 rounded-xl border border-white/20 overflow-hidden mb-4">
                    <img
                      src={extractedFaceUrl}
                      alt="Extracted face"
                      className="w-full h-full object-cover"
                    />
                    </div>

                    <button
                      onClick={handleDownload}
                className="w-full py-3 bg-white text-black rounded-lg hover:bg-white/90 transition-all font-medium text-sm flex items-center justify-center space-x-2"
                    >
                      <Download className="w-4 h-4" />
                <span>Download Face</span>
                    </button>
              </div>

              {/* Stats */}
              {extractionData && (
              <div className="bg-white/5 rounded-2xl border border-white/10 p-6">
                <h4 className="text-sm font-semibold text-white mb-4">Extraction Details</h4>
                
                <div className="grid grid-cols-2 gap-3 mb-4">
                  <div className="bg-white/5 p-3 rounded-lg">
                    <p className="text-xs text-white/50 mb-1">Faces Detected</p>
                    <p className="text-xl font-semibold text-white">{extractionData.totalFaces}</p>
                    </div>
                  <div className="bg-white/5 p-3 rounded-lg">
                    <p className="text-xs text-white/50 mb-1">Confidence</p>
                    <p className="text-xl font-semibold text-white">{extractionData.confidence}%</p>
                    </div>
                  </div>

                <div className="bg-white/5 p-3 rounded-lg text-xs">
                  <p className="text-white/50 mb-2">Selection Score</p>
                  <div className="space-y-1">
                    <div className="flex justify-between">
                      <span className="text-white/60">Size Score:</span>
                      <span className="text-white font-semibold">{extractionData.sizeScore}%</span>
                      </div>
                    <div className="flex justify-between">
                      <span className="text-white/60">Position Score:</span>
                      <span className="text-white font-semibold">{extractionData.positionScore}%</span>
                      </div>
                    </div>
                  </div>
                </div>
              )}

              {/* Actions */}
                <button
                  onClick={handleReset}
              className="w-full py-4 bg-white/10 text-white rounded-xl border border-white/20 hover:bg-white/20 font-medium flex items-center justify-center space-x-2 transition-all"
                >
                  <RefreshCw className="w-4 h-4" />
              <span>Capture Another ID</span>
                </button>
          </motion.div>
          )}

        {/* Hidden canvas for capture */}
        <canvas ref={canvasRef} className="hidden" />
      </div>
    </div>
  );
}
