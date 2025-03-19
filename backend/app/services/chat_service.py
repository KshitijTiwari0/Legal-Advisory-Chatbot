from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import Ollama
from transformers import AutoTokenizer
from app.config.settings import settings
from app.services.pdf_service import PDFService

class ChatService:
    def __init__(self):
        self.qa_chain = None
        self.source_text = None
        self.initialize_qa_system()

    def initialize_qa_system(self, recreate_index: bool = False):
        """Initialize the QA system."""
        self.source_text = PDFService.extract_text_from_pdf()
        chunks = self._create_chunks(self.source_text)
        
        embeddings = HuggingFaceEmbeddings(model_name=settings.EMBEDDING_MODEL)
        db = self._load_or_create_faiss_index(embeddings, chunks, recreate_index)
        
        llm = Ollama(
            model=settings.LLM_MODEL,
            temperature=settings.TEMPERATURE,
            callbacks=[StreamingStdOutCallbackHandler()]
        )
        
        self.qa_chain = ConversationalRetrievalChain.from_llm(
            llm,
            db.as_retriever(search_kwargs={"k": 3}),
            return_source_documents=True
        )

    def _create_chunks(self, text: str) -> list:
        """Split text into chunks."""
        tokenizer = AutoTokenizer.from_pretrained(settings.TOKENIZER_NAME)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            length_function=lambda x: len(tokenizer.encode(x)),
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        return text_splitter.split_text(text)

    def _load_or_create_faiss_index(self, embeddings, chunks, recreate_index: bool):
        """Load or create FAISS index."""
        if not recreate_index and os.path.exists(settings.FAISS_INDEX):
            return FAISS.load_local(settings.FAISS_INDEX, embeddings)
        else:
            db = FAISS.from_texts(chunks, embeddings)
            db.save_local(settings.FAISS_INDEX)
            return db

    def process_query(self, query: str, chat_history: list) -> str:
        """Process user queries."""
        if "what is this pdf about" in query.lower():
            return self._generate_document_summary()
        
        result = self.qa_chain({"question": query, "chat_history": chat_history})
        return result["answer"]

    def _generate_document_summary(self) -> str:
        """Generate a document summary."""
        summary_chunk = self.source_text[:3000]
        llm = Ollama(model=settings.LLM_MODEL, temperature=0.1)
        return llm(f"{settings.DOC_SUMMARY_PROMPT}\n\nText: {summary_chunk}") 
