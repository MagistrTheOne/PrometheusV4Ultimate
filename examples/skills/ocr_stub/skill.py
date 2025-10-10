"""OCR Stub skill implementation."""

import os
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from libs.skills import BaseSkill, SkillSpec, SkillRunResult, PermissionType, ResourceLimit


class OCRStubSkill(BaseSkill):
    """Stub OCR skill for text extraction from images."""
    
    def __init__(self):
        spec = SkillSpec(
            name="ocr_stub",
            version="1.0.0",
            description="Stub OCR implementation for text extraction from images",
            inputs={
                "input_file": "Path to image file for OCR processing",
                "output_file": "Path to save extracted text",
                "language": "Language for OCR: en, ru, auto (default: auto)",
                "confidence_threshold": "Minimum confidence for text detection (default: 0.5)"
            },
            outputs={
                "output_file": "Path to the text output file",
                "text_length": "Length of extracted text",
                "confidence_score": "Average confidence score",
                "processing_time": "Time taken for OCR processing",
                "image_format": "Format of input image"
            },
            perms={
                PermissionType.FS_READ: True,
                PermissionType.FS_WRITE: True,
                PermissionType.NETWORK: False,
                PermissionType.ENV_VAR: False
            },
            limits={
                ResourceLimit.CPU_MS: 10000,
                ResourceLimit.RAM_MB: 200,
                ResourceLimit.TIME_S: 60,
                ResourceLimit.DISK_MB: 100
            },
            tags=["ocr", "text", "image", "extraction", "stub"],
            author="PrometheusULTIMATE",
            license="MIT"
        )
        super().__init__(spec)
    
    def _execute(self, **kwargs) -> Dict[str, Any]:
        """Execute OCR processing."""
        input_file = kwargs["input_file"]
        output_file = kwargs["output_file"]
        language = kwargs.get("language", "auto")
        confidence_threshold = float(kwargs.get("confidence_threshold", 0.5))
        
        # Validate input file
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"Input file not found: {input_file}")
        
        # Get image format
        image_format = self._get_image_format(input_file)
        
        # Start timing
        start_time = time.time()
        
        # Simulate OCR processing
        extracted_text = self._simulate_ocr(input_file, language, confidence_threshold)
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Save extracted text
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(extracted_text)
        
        # Calculate metrics
        text_length = len(extracted_text)
        confidence_score = self._calculate_confidence_score(extracted_text, confidence_threshold)
        
        return {
            "output_file": output_file,
            "text_length": text_length,
            "confidence_score": confidence_score,
            "processing_time": round(processing_time, 3),
            "image_format": image_format
        }
    
    def _get_image_format(self, file_path: str) -> str:
        """Get image format from file extension."""
        path = Path(file_path)
        extension = path.suffix.lower()
        
        image_formats = {
            '.jpg': 'JPEG',
            '.jpeg': 'JPEG',
            '.png': 'PNG',
            '.gif': 'GIF',
            '.bmp': 'BMP',
            '.tiff': 'TIFF',
            '.tif': 'TIFF',
            '.webp': 'WebP'
        }
        
        return image_formats.get(extension, 'Unknown')
    
    def _simulate_ocr(self, input_file: str, language: str, confidence_threshold: float) -> str:
        """Simulate OCR text extraction."""
        
        # Get file name for context
        file_name = Path(input_file).stem
        
        # Simulate different text based on file name and language
        if "document" in file_name.lower():
            if language == "ru":
                return self._get_russian_document_text()
            else:
                return self._get_english_document_text()
        elif "receipt" in file_name.lower():
            return self._get_receipt_text()
        elif "business_card" in file_name.lower():
            return self._get_business_card_text()
        else:
            return self._get_generic_text(language)
    
    def _get_english_document_text(self) -> str:
        """Get simulated English document text."""
        return """PROMETHEUS ULTIMATE v4
Document Processing System

This is a simulated OCR result for an English document.
The system has successfully extracted text from the image.

Key Features:
- Advanced text recognition
- Multi-language support
- High accuracy processing
- Batch processing capabilities

For more information, visit our documentation.
"""
    
    def _get_russian_document_text(self) -> str:
        """Get simulated Russian document text."""
        return """ПРОМЕТЕЙ УЛЬТИМАТ v4
Система обработки документов

Это симулированный результат OCR для русского документа.
Система успешно извлекла текст из изображения.

Основные возможности:
- Продвинутое распознавание текста
- Поддержка нескольких языков
- Высокая точность обработки
- Возможности пакетной обработки

Для получения дополнительной информации посетите нашу документацию.
"""
    
    def _get_receipt_text(self) -> str:
        """Get simulated receipt text."""
        return """STORE RECEIPT
Date: 2024-01-15
Time: 14:30:25

Items:
1. Coffee - $3.50
2. Sandwich - $7.99
3. Cookie - $2.25

Subtotal: $13.74
Tax: $1.10
Total: $14.84

Payment: Credit Card
Card: ****1234

Thank you for your business!
"""
    
    def _get_business_card_text(self) -> str:
        """Get simulated business card text."""
        return """John Smith
Senior Software Engineer

PrometheusULTIMATE
123 Innovation Drive
Tech City, TC 12345

Phone: +1 (555) 123-4567
Email: john.smith@prometheus.com
Website: www.prometheus.com

LinkedIn: linkedin.com/in/johnsmith
"""
    
    def _get_generic_text(self, language: str) -> str:
        """Get generic simulated text."""
        if language == "ru":
            return """Это симулированный результат OCR.
Текст был извлечен из изображения с помощью
системы распознавания символов.

Дата обработки: 2024-01-15
Время: 14:30:25
"""
        else:
            return """This is a simulated OCR result.
Text has been extracted from the image using
optical character recognition system.

Processing date: 2024-01-15
Time: 14:30:25
"""
    
    def _calculate_confidence_score(self, text: str, threshold: float) -> float:
        """Calculate simulated confidence score."""
        
        # Simulate confidence based on text characteristics
        base_confidence = 0.8
        
        # Adjust based on text length
        if len(text) < 10:
            base_confidence -= 0.2
        elif len(text) > 100:
            base_confidence += 0.1
        
        # Adjust based on text quality indicators
        if any(char.isdigit() for char in text):
            base_confidence += 0.05  # Numbers are usually well recognized
        
        if any(char.isupper() for char in text):
            base_confidence += 0.05  # Uppercase letters are usually clear
        
        # Ensure confidence is within bounds
        confidence = max(0.0, min(1.0, base_confidence))
        
        # Apply threshold
        if confidence < threshold:
            confidence = threshold - 0.1  # Slightly below threshold
        
        return round(confidence, 3)
