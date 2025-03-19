import os

class Settings:
    PDF_PATH = os.getenv("PDF_PATH", "FR.pdf")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    LLM_MODEL = os.getenv("LLM_MODEL", "tinyllama")
    FAISS_INDEX = os.getenv("FAISS_INDEX", "faiss_index")
    TOKENIZER_NAME = os.getenv("TOKENIZER_NAME", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 512))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 64))
    TEMPERATURE = float(os.getenv("TEMPERATURE", 0.3))
    MIN_TEXT_LENGTH = int(os.getenv("MIN_TEXT_LENGTH", 100))

settings = Settings() 
