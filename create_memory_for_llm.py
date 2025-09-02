from langchain_community.document_loaders import PyPDFLoader,DirectoryLoader

from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

#step 1: Load Raw Data
DATA_PATH = "data/"

def load_pdf_files(data_path):
    loader = DirectoryLoader(data_path, 
                             glob="**/*.pdf", 
                             loader_cls=PyPDFLoader)
    return loader.load()

documents = load_pdf_files(DATA_PATH)
#print(len(documents), "pages loaded.")

# Step 2: Create Chunks
def create_chunks(documents, chunk_size=500, chunk_overlap=50):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return text_splitter.split_documents(documents)

text_chunks = create_chunks(documents)
# print(len(text_chunks), "chunks created.")

# Step 3: Create Embeddings
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


# Step 4: Store embeddings in FAISS
DB_FAISS_PATH = "vectorstore/db_faiss"
db = FAISS.from_documents(text_chunks, embedding_model)
db.save_local(DB_FAISS_PATH)

