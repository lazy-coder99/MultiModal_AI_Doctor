#set GROQ API 
from dotenv import load_dotenv
load_dotenv()
import os
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")


#step2: Convert Image to required format
import base64 #encoding using base64

def encode_image(image_path):
  if image_path is None:
    return None
  image_file = open(image_path,"rb")
  return base64.b64encode(image_file.read()).decode('utf-8')

#step3: Multimodal LLm setup

from groq import Groq

query="Is there something wrong with my face?"
#model = "meta-llama/llama-4-maverick-17b-128e-instruct"
model="meta-llama/llama-4-scout-17b-16e-instruct"
#model = "meta-llama/llama-4-scout-17b-16e-instruct"
#model="llama-3.2-90b-vision-preview" #Deprecated

#================================RAG================================
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

DB_FAISS_PATH = "vectorstore/db_faiss"
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local(DB_FAISS_PATH, embeddings=embedding_model, allow_dangerous_deserialization=True)
#====================================================================



def analyze_image_with_query(query, model, encoded_image=None,base_prompt=None):
    
    #=======rag retrieval========
    docs = db.similarity_search(query, k=3)
    retrieved_context = "\n".join([doc.page_content for doc in docs])

    full_prompt = f"""
    You are a helpful medical assistant. Answer the query as a doctor.

    User Query:
    {query}

    Below is some Retrieved Context if you find them useful for your answer then use them, otherwise ignore them.

    Retrieved Context:
    {retrieved_context}  
    
    """
    #========================================

    client = Groq()
    messages = [{"role": "user", "content": []},{"role": "system", "content":  str(base_prompt) if base_prompt else "You are a helpful medical assistant."}]
    
    # Add text query
    messages[0]["content"].append({"type": "text", "text": str(full_prompt)})
    
    # Add image if provided
    if encoded_image:
        messages[0]["content"].append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}
        })
    # print("!####\n",messages)
    response = client.chat.completions.create(
        messages=messages,
        model=model
    )
    return response.choices[0].message.content


# print(analyze_image_with_query(query, model, encode_image("acne.jpg")))