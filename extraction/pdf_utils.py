"""PDF utilities for splitting and manipulation."""

import io
from typing import List

from pypdf import PdfReader, PdfWriter


def split_pdf_into_chunks(pdf_bytes: bytes, *, pages_per_chunk: int = 20) -> List[bytes]:
    """Split a PDF into smaller PDFs, each containing pages_per_chunk pages.
    
    Args:
        pdf_bytes: Original PDF bytes
        pages_per_chunk: Maximum pages per chunk (default 20)
        
    Returns:
        List of PDF bytes, each representing a chunk
    """
    reader = PdfReader(io.BytesIO(pdf_bytes))
    total_pages = len(reader.pages)
    
    chunks: List[bytes] = []
    for start_idx in range(0, total_pages, pages_per_chunk):
        end_idx = min(start_idx + pages_per_chunk, total_pages)
        
        writer = PdfWriter()
        for page_idx in range(start_idx, end_idx):
            writer.add_page(reader.pages[page_idx])
        
        chunk_buffer = io.BytesIO()
        writer.write(chunk_buffer)
        chunk_buffer.seek(0)
        chunks.append(chunk_buffer.read())
    
    return chunks


def get_page_count(pdf_bytes: bytes) -> int:
    """Get the total number of pages in a PDF.
    
    Args:
        pdf_bytes: PDF file bytes
        
    Returns:
        Number of pages in the PDF
    """
    reader = PdfReader(io.BytesIO(pdf_bytes))
    return len(reader.pages)
