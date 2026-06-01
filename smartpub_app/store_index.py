from src.helper import load_PDF, text_split, download_hugging_face_embedding
from pinecone import Pinecone as pc
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
import os


load_dotenv()

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")


extracted_data = load_PDF("data/")
text_chunks = text_split(extracted_data)
embeddings = download_hugging_face_embedding()

pinecone_client = pc(api_key=PINECONE_API_KEY)

index_name = "medical-chatbot"
docsearch = PineconeVectorStore.from_texts(
    [t.page_content for t in text_chunks],
    embeddings,
    index_name=index_name,
)
