from sentence_transformers import SentenceTransformer
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
import google.generativeai as genai
from module import load_data, split_text_with_overlap, build_prompt
from dotenv import load_dotenv
import os
import warnings
import numpy as np

warnings.filterwarnings("ignore", category=FutureWarning)
load_dotenv()

# Load and split documents
loaded_data = load_data(["Report.pdf", "Research_paper.pdf", "new_cv.pdf"])

# Split documents into chunks
all_chunks = []
for doc in loaded_data:
    all_chunks.extend(split_text_with_overlap(doc))

# Load embedding model
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# Connect to Milvus
connections.connect("default", host="localhost", port="19530")

# Define collection schema
collection_name = "my_docs"
dim = 384  # Dimension for all-MiniLM-L6-v2 model

# Delete collection if it exists
if utility.has_collection(collection_name):
    utility.drop_collection(collection_name)

# Define fields
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim),
    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535)
]

# Create collection schema
schema = CollectionSchema(fields, "Document chunks collection")

# Create collection
collection = Collection(collection_name, schema)

# Prepare data for insertion
embeddings = []
texts = []

print(f"Processing {len(all_chunks)} chunks...")
for chunk in all_chunks:
    embedding = embed_model.encode(chunk)
    embeddings.append(embedding.tolist())
    texts.append(chunk)

# Insert data
entities = [
    embeddings,  # embedding field
    texts       # text field
]

insert_result = collection.insert(entities)
print(f"Inserted {len(insert_result.primary_keys)} entities")

# Create index for better search performance
index_params = {
    "metric_type": "COSINE",
    "index_type": "IVF_FLAT",
    "params": {"nlist": 128}
}

collection.create_index("embedding", index_params)

# Load collection to memory
collection.load()

# Query
question = "What are Cloud based project?"
query_embedding = embed_model.encode(question).tolist()

# Search parameters
search_params = {
    "metric_type": "COSINE",
    "params": {"nprobe": 10}
}

# Perform search
results = collection.search(
    data=[query_embedding],
    anns_field="embedding",
    param=search_params,
    limit=5,
    output_fields=["text"]
)

# Extract retrieved context
retrieved_context = "\n".join([hit.entity.get('text') for hit in results[0]])

# Build prompt
final_prompt = build_prompt(retrieved_context, question)

# Configure Gemini API
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
llm = genai.GenerativeModel("gemini-2.0-flash")

# Generate response
response = llm.generate_content(final_prompt)
print(response.text)

# Clean up
collection.release()
connections.disconnect("default")