"""
South African ID validation utilities.
Handles ID number validation, checksum calculation, and date validation.
"""

from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


def validate_sa_id_date(yy: str, mm: str, dd: str) -> bool:
    """
    Validate date portion of SA ID number.
    
    Args:
        yy: Year (2 digits)
        mm: Month (2 digits)
        dd: Day (2 digits)
    
    Returns:
        True if date is valid, False otherwise
    """
    try:
        year = int(yy)
        month = int(mm)
        day = int(dd)
        
        # Month validation
        if not (1 <= month <= 12):
            return False
        
        # Day validation (basic - doesn't check month-specific max days)
        if not (1 <= day <= 31):
            return False
        
        # Year validation (reasonable range: 1900-2099)
        # Years 00-24 are 2000-2024, 25-99 are 1925-1999
        current_year = 24  # 2024
        if year > current_year and year < 25:
            return False  # Future dates not realistic
        
        return True
    except:
        return False


def calculate_sa_id_checksum(id_12_digits: str) -> int:
    """
    Calculate SA ID checksum using Luhn algorithm.
    
    SA ID Format: YYMMDD SSSS C AZ
    - YYMMDD: Date of birth
    - SSSS: Sequence number (also encodes gender)
    - C: Citizenship (0=SA citizen, 1=permanent resident)
    - A: Usually 8 or 9
    - Z: Checksum digit
    
    Args:
        id_12_digits: First 12 digits of ID number
    
    Returns:
        Expected checksum digit (0-9), or -1 if calculation fails
    """
    try:
        # Luhn algorithm for SA ID
        digits = [int(d) for d in id_12_digits]
        
        # Step 1: Sum odd-position digits (1st, 3rd, 5th, etc.)
        odd_sum = sum(digits[i] for i in range(0, 12, 2))
        
        # Step 2: Concatenate even-position digits and multiply by 2
        even_concat = ''.join(str(digits[i]) for i in range(1, 12, 2))
        even_doubled = int(even_concat) * 2
        
        # Step 3: Sum all digits of the doubled result
        even_sum = sum(int(d) for d in str(even_doubled))
        
        # Step 4: Add odd_sum and even_sum
        total = odd_sum + even_sum
        
        # Step 5: Checksum is (10 - (total % 10)) % 10
        checksum = (10 - (total % 10)) % 10
        
        return checksum
    except Exception as e:
        logger.error(f"Checksum calculation error: {e}")
        return -1


def validate_sa_id_number(id_number: str) -> Tuple[bool, int]:
    """
    Validate SA ID number with comprehensive checks.
    
    Args:
        id_number: 13-digit SA ID number
    
    Returns:
        Tuple of (is_valid, score)
        - is_valid: Boolean indicating if ID is valid
        - score: Validation score (0-100)
    """
    if len(id_number) != 13 or not id_number.isdigit():
        return False, 0
    
    score = 0
    
    try:
        # Extract components
        yy = id_number[0:2]
        mm = id_number[2:4]
        dd = id_number[4:6]
        gender_digit = int(id_number[6])
        citizenship = int(id_number[10])
        checksum_digit = int(id_number[12])
        
        # 1. Date validation (weight: 30 points)
        if validate_sa_id_date(yy, mm, dd):
            score += 30
        else:
            return False, 0  # Invalid date
        
        # 2. Checksum validation (weight: 50 points)
        expected_checksum = calculate_sa_id_checksum(id_number[:12])
        if expected_checksum == checksum_digit:
            score += 50
        else:
            logger.info(f"ID {id_number}: Checksum mismatch (expected {expected_checksum}, got {checksum_digit})")
            return False, 0  # Invalid checksum
        
        # 3. Citizenship validation (weight: 10 points)
        if citizenship in [0, 1]:  # 0=SA citizen, 1=permanent resident
            score += 10
        
        # 4. Gender digit validation (weight: 10 points)
        if 0 <= gender_digit <= 9:  # Any digit is valid (0-4=F, 5-9=M)
            score += 10
        
        return True, score
    
    except Exception as e:
        logger.error(f"ID validation error: {e}")
        return False, 0


def extract_info_from_id(id_number: str) -> dict:
    """
    Extract information encoded in SA ID number.
    
    Args:
        id_number: Valid 13-digit SA ID number
    
    Returns:
        Dictionary with extracted info (birth_date, gender, citizenship)
    """
    if len(id_number) != 13:
        return {}
    
    try:
        # Extract components
        yy = id_number[0:2]
        mm = id_number[2:4]
        dd = id_number[4:6]
        gender_digit = int(id_number[6])
        citizenship_digit = int(id_number[10])
        
        # Determine century (00-24 = 2000s, 25-99 = 1900s)
        year = int(yy)
        century = '20' if year <= 24 else '19'
        birth_date = f"{dd}/{mm}/{century}{yy}"
        
        # Gender (digit 7: 0-4=Female, 5-9=Male)
        gender = 'M' if gender_digit >= 5 else 'F'
        
        # Citizenship
        citizenship = 'SA Citizen' if citizenship_digit == 0 else 'Permanent Resident'
        
        return {
            'birth_date': birth_date,
            'gender': gender,
            'citizenship': citizenship
        }
    
    except Exception as e:
        logger.error(f"Info extraction error: {e}")
        return {}

