"""
OCR extraction module for South African ID documents.
Handles text extraction, field parsing, and validation.
"""

import cv2
import numpy as np
import re
from typing import Dict, Any, List, Tuple, Optional
from paddleocr import PaddleOCR
import logging

from validators import validate_sa_id_number, extract_info_from_id

logger = logging.getLogger(__name__)

# Initialize PaddleOCR (will be set by init function)
ocr = None


def initialize_ocr():
    """Initialize PaddleOCR with custom model directory."""
    global ocr
    try:
        paddleocr_home = '/app/.paddleocr'
        
        ocr = PaddleOCR(
            use_angle_cls=True, 
            lang='en', 
            use_gpu=False, 
            show_log=False,
            rec_model_dir=f'{paddleocr_home}/rec',
            det_model_dir=f'{paddleocr_home}/det',
            cls_model_dir=f'{paddleocr_home}/cls'
        )
        logger.info("✅ PaddleOCR initialized successfully")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to initialize PaddleOCR: {str(e)}")
        return False


def preprocess_for_ocr(image: np.ndarray) -> np.ndarray:
    """
    Preprocess image for better OCR accuracy.
    
    Args:
        image: Input image in BGR format
    
    Returns:
        Preprocessed image optimized for OCR
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply denoising
    denoised = cv2.fastNlMeansDenoising(gray)
    
    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(
        denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    
    # Convert back to BGR for PaddleOCR
    processed = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
    
    return processed


def extract_sa_id_number(text: str) -> Optional[str]:
    """
    Extract and validate South African ID number from text.
    Uses multi-pattern matching, date validation, and checksum verification.
    
    Args:
        text: Raw OCR text
    
    Returns:
        Validated 13-digit ID number, or None if not found
    """
    # Clean text: remove common separators
    clean_text = text.replace(' ', '').replace('-', '').replace('/', '')
    
    # Pattern: 13 consecutive digits
    pattern = r'\b(\d{13})\b'
    matches = re.findall(pattern, clean_text)
    
    if not matches:
        # Try original text with spaces/dashes
        pattern_with_sep = r'\b(\d{6})[\s\-]?(\d{4})[\s\-]?(\d{2})[\s\-]?(\d{1})\b'
        sep_matches = re.findall(pattern_with_sep, text)
        if sep_matches:
            matches = [''.join(match) for match in sep_matches]
    
    # Validate and score each candidate
    candidates = []
    
    for id_number in matches:
        if len(id_number) != 13:
            continue
        
        is_valid, score = validate_sa_id_number(id_number)
        
        if is_valid:
            candidates.append((id_number, score))
    
    if candidates:
        # Return highest scoring candidate
        candidates.sort(key=lambda x: x[1], reverse=True)
        best_id = candidates[0][0]
        logger.info(f"✅ SA ID validated: {best_id} (score: {candidates[0][1]})")
        return best_id
    
    return None


def extract_text_with_ocr(image: np.ndarray) -> Tuple[List[Tuple], str]:
    """
    Extract text from image using PaddleOCR.
    
    Args:
        image: Input image in BGR format
    
    Returns:
        Tuple of (extracted_data, raw_text)
        - extracted_data: List of (text, confidence, box) tuples
        - raw_text: All extracted text concatenated
    """
    try:
        # Run OCR
        result = ocr.ocr(image, cls=True)
        
        if not result or not result[0]:
            return [], ""
        
        # Extract text and confidence
        extracted_data = []
        raw_text_lines = []
        
        for line in result[0]:
            box = line[0]  # Coordinates
            text_info = line[1]  # (text, confidence)
            text = text_info[0].strip()
            confidence = text_info[1]
            
            if confidence > 0.5:  # Only include confident results
                extracted_data.append((text, confidence, box))
                raw_text_lines.append(text)
        
        raw_text = ' '.join(raw_text_lines)
        logger.info(f"📝 OCR extracted {len(extracted_data)} text elements")
        
        return extracted_data, raw_text
        
    except Exception as e:
        logger.error(f"OCR error: {str(e)}")
        return [], ""


def parse_smart_id_card(ocr_data: List[Tuple], raw_text: str) -> Dict[str, Any]:
    """
    Parse Smart ID Card fields.
    
    Expected fields:
    - Surname
    - Names  
    - ID Number
    - Sex
    - Country of Birth
    - Status
    
    Args:
        ocr_data: List of (text, confidence, box) tuples
        raw_text: All text concatenated
    
    Returns:
        Dictionary with extracted fields
    """
    result = {
        'surname': {'value': None, 'confidence': 0, 'status': 'not_detected'},
        'names': {'value': None, 'confidence': 0, 'status': 'not_detected'},
        'idNumber': {'value': None, 'confidence': 0, 'status': 'not_detected'},
        'sex': {'value': None, 'confidence': 0, 'status': 'not_detected'},
        'countryOfBirth': {'value': None, 'confidence': 0, 'status': 'not_detected'},
        'status': {'value': None, 'confidence': 0, 'status': 'not_detected'},
    }
    
    # Extract ID number first (most reliable with validation)
    id_number = extract_sa_id_number(raw_text)
    if id_number:
        result['idNumber'] = {'value': id_number, 'confidence': 99, 'status': 'success'}
        
        # Extract additional info from ID number
        id_info = extract_info_from_id(id_number)
        if id_info.get('gender'):
            result['sex'] = {'value': id_info['gender'], 'confidence': 99, 'status': 'success'}
        
        logger.info(f"📅 Birth date: {id_info.get('birth_date')}, Gender: {id_info.get('gender')}")
    
    # Keywords for field detection
    surname_keywords = ['surname', 'van']
    names_keywords = ['names', 'name']
    country_keywords = ['country', 'birth', 'rsa', 'south africa', 'za']
    status_keywords = ['status', 'citizen', 'citn']
    
    for i, (text, confidence, box) in enumerate(ocr_data):
        text_lower = text.lower()
        
        # Surname detection
        if any(kw in text_lower for kw in surname_keywords) and i + 1 < len(ocr_data):
            if 'surname' in text_lower:
                next_text = ocr_data[i + 1][0]
                result['surname'] = {'value': next_text, 'confidence': int(confidence * 100), 'status': 'success'}
        
        # Names detection
        if any(kw in text_lower for kw in names_keywords) and i + 1 < len(ocr_data):
            if 'name' in text_lower and 'surname' not in text_lower:
                next_text = ocr_data[i + 1][0]
                result['names'] = {'value': next_text, 'confidence': int(confidence * 100), 'status': 'success'}
        
        # Country of birth
        if any(kw in text_lower for kw in country_keywords):
            if 'rsa' in text_lower or 'south africa' in text_lower or 'za' in text_lower:
                result['countryOfBirth'] = {'value': 'RSA', 'confidence': int(confidence * 100), 'status': 'success'}
            elif len(text) == 3 and text.isupper():
                result['countryOfBirth'] = {'value': text, 'confidence': int(confidence * 100), 'status': 'success'}
        
        # Status
        if any(kw in text_lower for kw in status_keywords):
            if 'citizen' in text_lower or 'citn' in text_lower:
                result['status'] = {'value': 'Citizen', 'confidence': int(confidence * 100), 'status': 'success'}
            else:
                result['status'] = {'value': text, 'confidence': int(confidence * 100), 'status': 'partial'}
    
    return result


def parse_green_id_book(ocr_data: List[Tuple], raw_text: str) -> Dict[str, Any]:
    """
    Parse Green ID Book fields.
    
    Expected fields:
    - ID Number
    - Surname
    - Names
    - Date Issued
    - Place of Birth
    
    Args:
        ocr_data: List of (text, confidence, box) tuples
        raw_text: All text concatenated
    
    Returns:
        Dictionary with extracted fields
    """
    result = {
        'idNumber': {'value': None, 'confidence': 0, 'status': 'not_detected'},
        'surname': {'value': None, 'confidence': 0, 'status': 'not_detected'},
        'names': {'value': None, 'confidence': 0, 'status': 'not_detected'},
        'dateIssued': {'value': None, 'confidence': 0, 'status': 'not_detected'},
        'placeOfBirth': {'value': None, 'confidence': 0, 'status': 'not_detected'},
    }
    
    # Extract ID number with validation
    id_number = extract_sa_id_number(raw_text)
    if id_number:
        result['idNumber'] = {'value': id_number, 'confidence': 99, 'status': 'success'}
    
    # Keywords
    surname_keywords = ['surname', 'van']
    names_keywords = ['names', 'name', 'voorname']
    date_keywords = ['issued', 'date', 'datum']
    place_keywords = ['place', 'birth', 'plek', 'geboorte']
    
    # Date pattern: DD/MM/YYYY or YYYY/MM/DD
    date_pattern = r'\b(\d{2}[/-]\d{2}[/-]\d{4}|\d{4}[/-]\d{2}[/-]\d{2})\b'
    
    for i, (text, confidence, box) in enumerate(ocr_data):
        text_lower = text.lower()
        
        # Surname
        if any(kw in text_lower for kw in surname_keywords) and i + 1 < len(ocr_data):
            next_text = ocr_data[i + 1][0]
            if not any(char.isdigit() for char in next_text):
                result['surname'] = {'value': next_text, 'confidence': int(confidence * 100), 'status': 'success'}
        
        # Names
        if any(kw in text_lower for kw in names_keywords) and i + 1 < len(ocr_data):
            if 'surname' not in text_lower:
                next_text = ocr_data[i + 1][0]
                if not any(char.isdigit() for char in next_text):
                    result['names'] = {'value': next_text, 'confidence': int(confidence * 100), 'status': 'success'}
        
        # Date issued
        if any(kw in text_lower for kw in date_keywords) or re.search(date_pattern, text):
            date_match = re.search(date_pattern, text)
            if date_match:
                result['dateIssued'] = {'value': date_match.group(0), 'confidence': int(confidence * 100), 'status': 'success'}
            elif i + 1 < len(ocr_data):
                next_text = ocr_data[i + 1][0]
                date_match = re.search(date_pattern, next_text)
                if date_match:
                    result['dateIssued'] = {'value': date_match.group(0), 'confidence': int(confidence * 100), 'status': 'success'}
        
        # Place of birth
        if any(kw in text_lower for kw in place_keywords) and i + 1 < len(ocr_data):
            next_text = ocr_data[i + 1][0]
            if not any(char.isdigit() for char in next_text):
                result['placeOfBirth'] = {'value': next_text, 'confidence': int(confidence * 100), 'status': 'success'}
    
    return result


def extract_id_data(image: np.ndarray, id_type: str) -> Dict[str, Any]:
    """
    Main OCR function that extracts data based on ID type.
    
    Args:
        image: Input image in BGR format
        id_type: 'smart', 'green', or 'full'
    
    Returns:
        Dictionary with extraction results
    """
    try:
        logger.info(f"🔍 Starting OCR for {id_type} ID...")
        
        # Preprocess image
        processed_image = preprocess_for_ocr(image)
        
        # Extract text
        ocr_data, raw_text = extract_text_with_ocr(processed_image)
        
        if not ocr_data:
            logger.warning("⚠️ No text detected by OCR")
            return {
                'success': False,
                'idType': id_type,
                'fields': {},
                'rawText': '',
                'confidence': 0,
                'message': 'No text detected in the image'
            }
        
        # Parse based on ID type
        if id_type == 'smart':
            fields = parse_smart_id_card(ocr_data, raw_text)
        elif id_type == 'green':
            fields = parse_green_id_book(ocr_data, raw_text)
        else:
            # Try both and see which one finds more fields
            smart_fields = parse_smart_id_card(ocr_data, raw_text)
            green_fields = parse_green_id_book(ocr_data, raw_text)
            
            # Count successful extractions
            smart_count = sum(1 for f in smart_fields.values() if f['status'] == 'success')
            green_count = sum(1 for f in green_fields.values() if f['status'] == 'success')
            
            if smart_count >= green_count:
                fields = smart_fields
                id_type = 'smart'
            else:
                fields = green_fields
                id_type = 'green'
        
        # Calculate overall confidence
        confidences = [f['confidence'] for f in fields.values() if f['confidence'] > 0]
        overall_confidence = sum(confidences) / len(confidences) if confidences else 0
        
        # Count field statuses
        success_count = sum(1 for f in fields.values() if f['status'] == 'success')
        total_fields = len(fields)
        
        logger.info(f"✅ OCR complete: {success_count}/{total_fields} fields extracted")
        
        return {
            'success': True,
            'idType': id_type,
            'fields': fields,
            'rawText': raw_text,
            'confidence': round(overall_confidence, 1),
            'fieldsExtracted': success_count,
            'totalFields': total_fields
        }
        
    except Exception as e:
        logger.error(f"❌ OCR extraction error: {str(e)}")
        return {
            'success': False,
            'idType': id_type,
            'fields': {},
            'rawText': '',
            'confidence': 0,
            'message': f'OCR failed: {str(e)}'
        }

