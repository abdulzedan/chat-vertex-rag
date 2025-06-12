import os
import tempfile
from typing import BinaryIO, Optional
import PyPDF2
from PIL import Image
import csv
import io
import logging

logger = logging.getLogger(__name__)

class DocumentProcessor:
    """Process different document types for upload"""
    
    @staticmethod
    async def process_pdf(file: BinaryIO) -> tuple[str, str]:
        """Extract text from PDF"""
        try:
            pdf_reader = PyPDF2.PdfReader(file)
            text_content = []
            
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text_content.append(f"Page {page_num + 1}:\n{page.extract_text()}")
            
            full_text = "\n\n".join(text_content)
            return full_text, "application/pdf"
        except Exception as e:
            logger.error(f"Error processing PDF: {e}")
            raise
    
    @staticmethod
    async def process_image(file: BinaryIO) -> tuple[str, str]:
        """Process image file (returns path for Gemini vision)"""
        try:
            # Save image temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
                image = Image.open(file)
                image.save(tmp_file.name, "PNG")
                return tmp_file.name, "image/png"
        except Exception as e:
            logger.error(f"Error processing image: {e}")
            raise
    
    @staticmethod
    async def process_csv(file: BinaryIO) -> tuple[str, str]:
        """Convert CSV to text format"""
        try:
            # Read CSV content
            content = file.read().decode('utf-8')
            file.seek(0)
            
            # Parse CSV
            csv_reader = csv.reader(io.StringIO(content))
            rows = list(csv_reader)
            
            if not rows:
                return "Empty CSV file", "text/csv"
            
            # Create a text representation
            text_parts = []
            text_parts.append(f"CSV Data Summary:")
            text_parts.append(f"Columns: {', '.join(rows[0])}")
            text_parts.append(f"Rows: {len(rows) - 1}")
            text_parts.append("\nData Preview:")
            
            # Show first 10 rows
            for i, row in enumerate(rows[:11]):  # Header + 10 data rows
                text_parts.append(" | ".join(row))
                if i == 0:  # After header
                    text_parts.append("-" * 50)
            
            if len(rows) > 11:
                text_parts.append("... (showing first 10 rows)")
            
            return "\n".join(text_parts), "text/csv"
        except Exception as e:
            logger.error(f"Error processing CSV: {e}")
            raise

async def save_uploaded_file(file_content: bytes, filename: str) -> str:
    """Save uploaded file to temporary location"""
    upload_dir = "uploads"
    os.makedirs(upload_dir, exist_ok=True)
    
    file_path = os.path.join(upload_dir, filename)
    
    with open(file_path, "wb") as f:
        f.write(file_content)
    
    return file_path