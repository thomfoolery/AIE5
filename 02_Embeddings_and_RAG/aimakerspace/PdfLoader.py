import PyPDF2
from typing import List, Dict, Tuple
from pathlib import Path

class PDFLoader:
    def __init__(self, path: str):
        self.path = Path(path)
        self.documents = []
        self.metadata = []
        
    def load_pdf(self) -> Tuple[List[str], List[Dict]]:
        """
        Loads a PDF file and returns a list of text chunks and their metadata
        """
        # Open PDF file
        with open(self.path, 'rb') as file:
            # Create PDF reader object
            pdf_reader = PyPDF2.PdfReader(file)
            
            # Extract text from each page
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text = page.extract_text()
                
                # Skip empty pages
                if text.strip():
                    self.documents.append(text)
                    # Store metadata for each chunk
                    self.metadata.append({
                        "source": str(self.path),
                        "page": page_num + 1,
                        "total_pages": len(pdf_reader.pages)
                    })
        
        return self.documents, self.metadata