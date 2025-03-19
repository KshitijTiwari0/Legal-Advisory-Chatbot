import PyPDF2
from app.config.settings import settings
from app.utils.validation import validate_pdf_path

class PDFService:
    @staticmethod
    def extract_text_from_pdf(pdf_path: str = settings.PDF_PATH) -> str:
        """Extract text from PDF."""
        validate_pdf_path(pdf_path)
        
        text = ""
        try:
            with open(pdf_path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
        except PyPDF2.errors.PdfReadError as e:
            raise RuntimeError(f"PDF reading error: {str(e)}")
        
        if len(text.strip()) < settings.MIN_TEXT_LENGTH:
            print("Warning: Low text extraction. Consider using OCR.")
        
        return text 
