# Copyright (C) 2025 FluxBoard Team
#
# This library is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation; either
# version 2.1 of the License, or (at your option) any later version.
#
# This library is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with this library; if not, write to the Free Software
# Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA

"""
Input validation utilities for OptixLog SDK

This module provides validation functions to catch common errors before API calls.
"""

import os
import math
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path


class ValidationError(ValueError):
    """Custom exception for validation errors"""
    pass


def validate_metrics(metrics: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """
    Validate metrics dictionary for common issues
    
    Args:
        metrics: Dictionary of metric name -> value
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not isinstance(metrics, dict):
        return False, f"Metrics must be a dictionary, got {type(metrics).__name__}"
    
    if not metrics:
        return False, "Metrics dictionary is empty"
    
    for key, value in metrics.items():
        # Check key is string
        if not isinstance(key, str):
            return False, f"Metric key must be string, got {type(key).__name__} for key {key}"
        
        # Check for numeric types
        if isinstance(value, (int, float)):
            # Check for NaN
            if math.isnan(value):
                return False, f"Metric '{key}' contains NaN. Use a default value or filter it out."
            
            # Check for Inf
            if math.isinf(value):
                return False, f"Metric '{key}' contains Inf. Consider capping large values."
            
        # Check for None
        elif value is None:
            return False, f"Metric '{key}' is None. Use 0 or remove the metric."
        
        # Allow strings, bools, lists, dicts
        elif not isinstance(value, (str, bool, list, dict)):
            return False, f"Metric '{key}' has unsupported type {type(value).__name__}"
    
    return True, None


def validate_file_path(file_path: str) -> Tuple[bool, Optional[str]]:
    """
    Validate that a file exists and is readable
    
    Args:
        file_path: Path to file
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not isinstance(file_path, (str, Path)):
        return False, f"File path must be string or Path, got {type(file_path).__name__}"
    
    path = Path(file_path)
    
    if not path.exists():
        return False, f"File not found: {file_path}"
    
    if not path.is_file():
        return False, f"Path is not a file: {file_path}"
    
    if not os.access(path, os.R_OK):
        return False, f"File is not readable: {file_path}"
    
    # Check file size (warn if > 100MB)
    file_size = path.stat().st_size
    if file_size > 100 * 1024 * 1024:  # 100MB
        return False, f"File too large ({file_size / 1024 / 1024:.1f} MB). Consider compressing or splitting."
    
    return True, None


def validate_image(image_obj: Any) -> Tuple[bool, Optional[str]]:
    """
    Validate PIL Image object
    
    Args:
        image_obj: PIL Image object
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        from PIL import Image
    except ImportError:
        return False, "PIL/Pillow not installed. Install with: pip install pillow"
    
    if not isinstance(image_obj, Image.Image):
        return False, f"Expected PIL Image, got {type(image_obj).__name__}"
    
    # Check image dimensions
    width, height = image_obj.size
    if width == 0 or height == 0:
        return False, f"Image has zero dimensions: {width}x{height}"
    
    if width > 10000 or height > 10000:
        return False, f"Image too large: {width}x{height}. Consider resizing to under 10000x10000."
    
    return True, None


def validate_step(step: Any) -> Tuple[bool, Optional[str]]:
    """
    Validate step parameter
    
    Args:
        step: Step number
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not isinstance(step, int):
        return False, f"Step must be an integer, got {type(step).__name__}"
    
    if step < 0:
        return False, f"Step must be non-negative, got {step}"
    
    return True, None


def validate_key(key: str) -> Tuple[bool, Optional[str]]:
    """
    Validate key/name parameter
    
    Args:
        key: Key string
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not isinstance(key, str):
        return False, f"Key must be a string, got {type(key).__name__}"
    
    if not key or key.strip() == "":
        return False, "Key cannot be empty"
    
    if len(key) > 200:
        return False, f"Key too long ({len(key)} chars). Keep it under 200 characters."
    
    return True, None


def sanitize_metrics(metrics: Dict[str, Any], 
                     nan_replacement: float = 0.0,
                     inf_replacement: Optional[float] = None,
                     remove_invalid: bool = False) -> Dict[str, Any]:
    """
    Sanitize metrics by replacing or removing invalid values
    
    Args:
        metrics: Dictionary of metrics
        nan_replacement: Value to replace NaN with (default: 0.0)
        inf_replacement: Value to replace Inf with (default: None, keeps as-is)
        remove_invalid: If True, remove invalid entries instead of replacing
        
    Returns:
        Sanitized metrics dictionary
    """
    sanitized = {}
    
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            if math.isnan(value):
                if remove_invalid:
                    continue
                value = nan_replacement
            elif math.isinf(value) and inf_replacement is not None:
                if remove_invalid:
                    continue
                value = inf_replacement
        
        sanitized[key] = value
    
    return sanitized


def guess_content_type(file_path: str) -> str:
    """
    Guess content type from file extension
    
    Args:
        file_path: Path to file
        
    Returns:
        MIME type string
    """
    ext = Path(file_path).suffix.lower()
    
    content_types = {
        '.png': 'image/png',
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.gif': 'image/gif',
        '.bmp': 'image/bmp',
        '.svg': 'image/svg+xml',
        '.webp': 'image/webp',
        '.csv': 'text/csv',
        '.txt': 'text/plain',
        '.json': 'application/json',
        '.xml': 'application/xml',
        '.pdf': 'application/pdf',
        '.zip': 'application/zip',
        '.tar': 'application/x-tar',
        '.gz': 'application/gzip',
        '.mp4': 'video/mp4',
        '.avi': 'video/x-msvideo',
        '.mov': 'video/quicktime',
        '.h5': 'application/x-hdf5',
        '.hdf5': 'application/x-hdf5',
        '.npy': 'application/octet-stream',
        '.npz': 'application/zip',
    }
    
    return content_types.get(ext, 'application/octet-stream')


def validate_api_key(api_key: Optional[str]) -> Tuple[bool, Optional[str]]:
    """
    Validate API key format
    
    Args:
        api_key: API key string
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not api_key:
        return False, "API key is required. Set OPTIX_API_KEY environment variable or pass api_key parameter."
    
    if not isinstance(api_key, str):
        return False, f"API key must be string, got {type(api_key).__name__}"
    
    if len(api_key) < 10:
        return False, "API key seems too short. Check your key at https://optixlog.com"
    
    return True, None


def validate_batch_size(batch_size: int, max_size: int = 1000) -> Tuple[bool, Optional[str]]:
    """
    Validate batch size
    
    Args:
        batch_size: Number of items in batch
        max_size: Maximum allowed batch size
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not isinstance(batch_size, int):
        return False, f"Batch size must be integer, got {type(batch_size).__name__}"
    
    if batch_size <= 0:
        return False, f"Batch size must be positive, got {batch_size}"
    
    if batch_size > max_size:
        return False, f"Batch size too large ({batch_size}). Maximum is {max_size}."
    
    return True, None

