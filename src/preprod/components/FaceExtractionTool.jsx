'use client';

import { useState } from 'react';
import { Upload, Loader2, AlertCircle, CheckCircle, RefreshCw, Download } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import { extractFaceFromDocument } from '@/lib/faceExtractionApi';

export default function FaceExtractionTool() {
  const [idFile, setIdFile] = useState(null);
  const [idPreviewUrl, setIdPreviewUrl] = useState(null);
  const [extractedFaceUrl, setExtractedFaceUrl] = useState(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [error, setError] = useState(null);
  const [extractionData, setExtractionData] = useState(null);

  const handleFileUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    setIdFile(file);
    setError(null);
    setExtractedFaceUrl(null);
    setExtractionData(null);

    const url = URL.createObjectURL(file);
    setIdPreviewUrl(url);

    await extractFace(file);
  };

  const extractFace = async (file) => {
    setIsProcessing(true);
    setError(null);

    try {
      console.log('🚀 Calling face extraction API...');
      
      // Call the HuggingFace API
      const result = await extractFaceFromDocument(file);

      if (!result.success) {
        throw new Error('Face extraction failed');
      }

      // Set the extracted face (already base64 encoded)
      setExtractedFaceUrl(result.faceImage);

      // Set metadata
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

      console.log('✅ Face extraction complete!');

    } catch (err) {
      console.error('❌ Error:', err);
      setError(err.message || 'Face extraction failed. Please try again.');
    } finally {
      setIsProcessing(false);
    }
  };

  const handleReset = () => {
    setIdFile(null);
    setExtractedFaceUrl(null);
    setExtractionData(null);
    setError(null);
    if (idPreviewUrl) URL.revokeObjectURL(idPreviewUrl);
    if (extractedFaceUrl) URL.revokeObjectURL(extractedFaceUrl);
    setIdPreviewUrl(null);
  };

  const handleDownload = () => {
    if (!extractedFaceUrl) return;
    
    // Create download link for base64 image
    const a = document.createElement('a');
    a.href = extractedFaceUrl;
    a.download = `extracted_face_${Date.now()}.jpg`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  };

  return (
    <div className="min-h-screen bg-black text-white py-12 px-4" style={{ fontFamily: 'JetBrains Mono, monospace' }}>
      <div className="max-w-6xl mx-auto">
        
        {/* Header */}
        <div className="mb-12">
          <h1 className="text-3xl font-bold text-white tracking-tight mb-2">
            Face Extraction
          </h1>
          <p className="text-white/60 font-light">
            Upload an ID document to extract and analyze the face photo
          </p>
          <div className="h-0.5 w-24 bg-gradient-to-r from-white to-transparent mt-4"></div>
        </div>

        {/* Error */}
        <AnimatePresence>
          {error && (
            <motion.div
              initial={{ opacity: 0, y: -10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0 }}
              className="mb-8 p-5 bg-red-500/10 border border-red-500/30 rounded-xl"
            >
              <div className="flex items-start space-x-3">
                <AlertCircle className="h-5 w-5 text-red-400 flex-shrink-0 mt-0.5" />
                <div>
                  <p className="text-sm font-semibold text-red-400 mb-1">Extraction Failed</p>
                  <p className="text-sm text-red-300/90">{error}</p>
                </div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Main Card */}
        <div className="bg-white/5 rounded-2xl border border-white/10 shadow-2xl overflow-hidden backdrop-blur-sm">
          
          {/* Upload Section */}
          {!idFile && (
            <div className="p-16">
              <label className="cursor-pointer block">
                <div className="border-2 border-dashed border-white/20 rounded-xl p-20 hover:border-white/40 hover:bg-white/5 transition-all">
                  <Upload className="w-20 h-20 text-white/40 mx-auto mb-8" />
                  <div className="text-center space-y-3">
                    <h3 className="text-xl font-semibold text-white">
                      Upload ID Document
                    </h3>
                    <p className="text-sm text-white/60 font-light">
                      Supported formats: JPEG, PNG, PDF (max 10MB)
                    </p>
                    <div className="pt-6">
                      <div className="inline-flex items-center space-x-2 px-8 py-3 bg-white text-black rounded-lg hover:bg-white/90 transition-colors font-medium text-sm">
                        <span>Choose File</span>
                      </div>
                    </div>
                  </div>
                </div>
                <input
                  type="file"
                  accept=".jpg,.jpeg,.png,.pdf"
                  onChange={handleFileUpload}
                  className="hidden"
                />
              </label>
            </div>
          )}

          {/* Processing */}
          {isProcessing && (
            <div className="p-20 text-center">
              <Loader2 className="w-16 h-16 text-white animate-spin mx-auto mb-8" />
              <div className="space-y-2">
                <h3 className="text-lg font-semibold text-white">Processing</h3>
                <p className="text-sm text-white/60 font-light">Detecting and extracting face from ID document...</p>
              </div>
            </div>
          )}

          {/* Results */}
          {idFile && !isProcessing && extractedFaceUrl && (
            <div>
              {/* Images */}
              <div className="grid grid-cols-1 md:grid-cols-2 divide-y md:divide-y-0 md:divide-x divide-white/10">
                
                {/* Original ID */}
                <div className="p-8">
                  <h4 className="text-sm font-semibold text-white mb-4 flex items-center">
                    <span className="w-6 h-6 bg-white/10 text-white rounded flex items-center justify-center text-xs mr-2">1</span>
                    Original Document
                  </h4>
                  <div className="relative aspect-video bg-white/5 rounded-lg border border-white/10 overflow-hidden">
                    <img
                      src={idPreviewUrl}
                      alt="Original ID"
                      className="w-full h-full object-contain"
                    />
                  </div>
                  <p className="text-xs text-white/40 mt-3 font-light">
                    {idFile.name} • {(idFile.size / 1024).toFixed(0)} KB
                  </p>
                </div>

                {/* Extracted Face */}
                <div className="p-8">
                  <h4 className="text-sm font-semibold text-white mb-4 flex items-center">
                    <span className="w-6 h-6 bg-white/10 text-white rounded flex items-center justify-center text-xs mr-2">2</span>
                    Extracted Face
                  </h4>
                  <div className="relative aspect-square max-w-sm mx-auto bg-white/5 rounded-lg border border-white/20 overflow-hidden">
                    <img
                      src={extractedFaceUrl}
                      alt="Extracted face"
                      className="w-full h-full object-cover"
                    />
                    <div className="absolute top-3 right-3">
                      <CheckCircle className="w-6 h-6 text-white bg-black/50 rounded-full backdrop-blur-sm" />
                    </div>
                  </div>
                  <div className="mt-6 text-center">
                    <button
                      onClick={handleDownload}
                      className="inline-flex items-center space-x-2 px-6 py-2.5 bg-white text-black rounded-lg hover:bg-white/90 transition-all font-medium text-sm"
                    >
                      <Download className="w-4 h-4" />
                      <span>Download</span>
                    </button>
                  </div>
                </div>
              </div>

              {/* Stats */}
              {extractionData && (
                <div className="border-t border-white/10 p-8 bg-black/20">
                  <h4 className="text-sm font-semibold text-white mb-6">
                    Extraction Details
                  </h4>
                  
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
                    <div className="bg-white/5 p-4 rounded-lg border border-white/10">
                      <p className="text-xs text-white/50 mb-2 font-light">Faces Detected</p>
                      <p className="text-2xl font-semibold text-white">{extractionData.totalFaces}</p>
                    </div>
                    <div className="bg-white/5 p-4 rounded-lg border border-white/10">
                      <p className="text-xs text-white/50 mb-2 font-light">Confidence</p>
                      <p className="text-2xl font-semibold text-white">{extractionData.confidence}%</p>
                    </div>
                    <div className="bg-white/5 p-4 rounded-lg border border-white/10">
                      <p className="text-xs text-white/50 mb-2 font-light">Face Size</p>
                      <p className="text-lg font-semibold text-white">{extractionData.faceSize.width}×{extractionData.faceSize.height}</p>
                    </div>
                    <div className="bg-white/5 p-4 rounded-lg border border-white/10">
                      <p className="text-xs text-white/50 mb-2 font-light">Selection Score</p>
                      <p className="text-2xl font-semibold text-white">{extractionData.combinedScore}%</p>
                    </div>
                  </div>

                  {/* Algorithm Details */}
                  <div className="bg-white/5 p-5 rounded-lg border border-white/10">
                    <p className="text-xs font-semibold text-white mb-4">Selection Algorithm</p>
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-xs font-light">
                      <div>
                        <span className="text-white/50">Size Score:</span>
                        <span className="ml-2 font-semibold text-white">{extractionData.sizeScore}%</span>
                        <span className="ml-1 text-white/30">(70% weight)</span>
                      </div>
                      <div>
                        <span className="text-white/50">Position Score:</span>
                        <span className="ml-2 font-semibold text-white">{extractionData.positionScore}%</span>
                        <span className="ml-1 text-white/30">(30% weight)</span>
                      </div>
                      <div>
                        <span className="text-white/50">Selected Face:</span>
                        <span className="ml-2 font-semibold text-white">#{extractionData.selectedIndex + 1}</span>
                        {extractionData.totalFaces > 1 && (
                          <span className="ml-1 text-white/50">(of {extractionData.totalFaces})</span>
                        )}
                      </div>
                    </div>
                  </div>

                  {/* Console Tip */}
                  <div className="mt-6 p-4 bg-blue-500/10 border border-blue-500/20 rounded-lg text-xs text-blue-300">
                    <span className="font-semibold">Tip:</span> Check the browser console (F12) for detailed extraction logs
                  </div>
                </div>
              )}

              {/* Actions */}
              <div className="border-t border-white/10 p-8 text-center bg-black/20">
                <button
                  onClick={handleReset}
                  className="inline-flex items-center space-x-2 px-6 py-2.5 bg-white/10 text-white rounded-lg border border-white/20 hover:bg-white/20 hover:border-white/30 font-medium text-sm transition-all"
                >
                  <RefreshCw className="w-4 h-4" />
                  <span>Upload Another ID</span>
                </button>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

