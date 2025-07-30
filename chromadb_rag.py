from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import google.generativeai as genai
from module import load_data, split_text_with_overlap, build_prompt
from dotenv import load_dotenv
import os

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

load_dotenv()

loaded_data = load_data(["PDF_Path1.pdf", "PDF_Path2.pdf", "so_on"]) #change the paths of pdf

#Split
all_chunks = []
for doc in loaded_data:
    all_chunks.extend(split_text_with_overlap(doc))

# Load embedding model
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# Set up ChromaDB
client = chromadb.Client(Settings(anonymized_telemetry=False))

# (Re)create collection
try:
    client.delete_collection(name="my_docs")

except:
    pass

collection = client.create_collection(name="my_docs")

# Embed and store
for idx, chunk in enumerate(all_chunks):
    embedding = embed_model.encode(chunk).tolist()
    collection.add(
        documents=[chunk],
        ids=[str(idx)],
        embeddings=[embedding])


# Query
question = input("enter your question")
query_embedding = embed_model.encode(question).tolist()


results = collection.query(query_embeddings=[query_embedding], n_results=5)
retrieved_context = "\n".join(doc for docs in results["documents"] for doc in docs)

# Build prompt
final_prompt = build_prompt(retrieved_context, question)

# Configure Gemini API
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
llm = genai.GenerativeModel("gemini-2.0-flash")

# Generate response
response = llm.generate_content(final_prompt)
print(response.text)


