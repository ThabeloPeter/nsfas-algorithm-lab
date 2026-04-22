/**
 * Face Extraction API Client
 * Handles communication with the HuggingFace Spaces API
 */

const API_URL = process.env.NEXT_PUBLIC_HF_FACE_API_URL || 'http://localhost:7860';
const TIMEOUT_MS = 30000;

const fetchWithTimeout = async (url, options = {}, timeout = TIMEOUT_MS) => {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeout);

  try {
    const response = await fetch(url, {
      ...options,
      signal: controller.signal,
    });
    clearTimeout(timeoutId);
    return response;
  } catch (error) {
    clearTimeout(timeoutId);
    if (error.name === 'AbortError') {
      throw new Error('Request timeout. The server took too long to respond.');
    }
    throw error;
  }
};

const readJsonWithTimeout = async (response, timeout = TIMEOUT_MS) => {
  let timer;
  const timeoutPromise = new Promise((_, reject) => {
    timer = setTimeout(() => {
      reject(new Error('The server took too long to finish sending a response.'));
    }, timeout);
  });

  try {
    return await Promise.race([response.json(), timeoutPromise]);
  } finally {
    clearTimeout(timer);
  }
};

const readErrorMessage = async (response) => {
  const jsonClone = response.clone();
  const textClone = response.clone();

  const normalizeMessage = (detail) => {
    if (!detail) {
      return `Server error: ${response.status}`;
    }
    if (typeof detail === 'string') {
      return detail;
    }
    if (typeof detail === 'object') {
      return detail.message || detail.error || detail.detail || JSON.stringify(detail);
    }
    return String(detail);
  };

  try {
    const data = await readJsonWithTimeout(jsonClone, 8000);
    return normalizeMessage(data?.detail || data?.error || data?.message);
  } catch {
    try {
      const text = await textClone.text();
      return text?.trim() || `Server error: ${response.status}`;
    } catch {
      return `Server error: ${response.status}`;
    }
  }
};

/**
 * Extract face from an ID document (image or PDF)
 * @param {File} file - The ID document file to process
 * @param {string} idType - 'smart', 'green', or 'full' for ROI optimization
 * @returns {Promise<Object>} Extracted face data and metadata
 */
export const extractFaceFromDocument = async (file, idType = 'full') => {
  try {
    if (!file) {
      throw new Error('No file provided');
    }

    const allowedTypes = ['image/jpeg', 'image/png', 'image/jpg', 'application/pdf'];
    if (!allowedTypes.includes(file.type)) {
      throw new Error('Invalid file type. Please upload JPEG, PNG, or PDF.');
    }

    const maxSize = 10 * 1024 * 1024;
    if (file.size > maxSize) {
      throw new Error('File too large. Maximum size is 10MB.');
    }

    const formData = new FormData();
    formData.append('file', file);
    formData.append('id_type', idType);

    console.log('Sending request to face extraction API...');
    console.log('File:', file.name, `(${(file.size / 1024).toFixed(2)} KB)`);
    console.log('ID Type:', idType);

    const response = await fetchWithTimeout(`${API_URL}/extract-face`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const errorMessage = await readErrorMessage(response);
      throw new Error(errorMessage);
    }

    const data = await readJsonWithTimeout(response);

    if (!data.success) {
      throw new Error(data.error || data.detail || 'Face extraction failed');
    }

    console.log('Face extraction successful');
    console.log('Metadata:', data.metadata);
    console.log('OCR Data:', data.ocr_data);

    return {
      success: true,
      faceImage: data.face_image,
      metadata: data.metadata,
      ocrData: data.ocr_data,
    };
  } catch (error) {
    console.error('Face extraction error:', error);

    if (error.name === 'AbortError' || error.message.toLowerCase().includes('timeout')) {
      throw new Error('The document capture took too long to process. Please try a clearer image or retry.');
    }

    if (error.message.includes('fetch')) {
      throw new Error('Unable to connect to face extraction service. Please check your internet connection.');
    }

    if (error.message.includes('No face detected')) {
      throw new Error('No face detected in the document. Please upload a clearer ID photo.');
    }

    throw error;
  }
};

/**
 * Compare two face images (selfie vs ID photo)
 * @param {File|Blob} selfieFile - The selfie image
 * @param {File|Blob} idPhotoFile - The ID photo image
 * @returns {Promise<Object>} Comparison result with match status and confidence
 */
export const compareFaces = async (selfieFile, idPhotoFile) => {
  try {
    if (!selfieFile || !idPhotoFile) {
      throw new Error('Both selfie and ID photo are required');
    }

    const formData = new FormData();
    formData.append('selfie', selfieFile);
    formData.append('id_photo', idPhotoFile);

    console.log('Sending face comparison request...');

    const response = await fetchWithTimeout(`${API_URL}/compare-faces`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const errorMessage = await readErrorMessage(response);
      throw new Error(errorMessage);
    }

    const data = await readJsonWithTimeout(response);

    if (!data.success) {
      throw new Error(data.error || data.detail || 'Face comparison failed');
    }

    console.log('Face comparison complete');
    console.log('Match:', data.match, '| Confidence:', data.confidence + '%');

    return {
      success: true,
      match: data.match,
      distance: data.distance,
      confidence: data.confidence,
      threshold: data.threshold,
    };
  } catch (error) {
    console.error('Face comparison error:', error);

    if (error.name === 'AbortError' || error.message.toLowerCase().includes('timeout')) {
      throw new Error('The selfie comparison took too long to finish. Please try again.');
    }

    if (error.message.includes('fetch')) {
      throw new Error('Unable to connect to face comparison service. Please check your internet connection.');
    }

    throw error;
  }
};

/**
 * Convert base64 image to Blob
 * @param {string} base64 - Base64 encoded image with data URL prefix
 * @returns {Blob} Image blob
 */
export const base64ToBlob = (base64) => {
  const parts = base64.split(',');
  const contentType = parts[0].match(/:(.*?);/)[1];
  const raw = window.atob(parts[1]);
  const rawLength = raw.length;
  const uint8Array = new Uint8Array(rawLength);

  for (let i = 0; i < rawLength; i++) {
    uint8Array[i] = raw.charCodeAt(i);
  }

  return new Blob([uint8Array], { type: contentType });
};

/**
 * Check API health status
 * @returns {Promise<Object>} API health status
 */
export const checkApiHealth = async () => {
  try {
    const response = await fetchWithTimeout(`${API_URL}/`, {
      method: 'GET',
    }, 5000);

    if (!response.ok) {
      return { healthy: false, error: `HTTP ${response.status}` };
    }

    const data = await response.json();
    return { healthy: true, ...data };
  } catch (error) {
    return { healthy: false, error: error.message };
  }
};

export default {
  extractFaceFromDocument,
  compareFaces,
  base64ToBlob,
  checkApiHealth,
};
