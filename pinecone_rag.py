from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
import google.generativeai as genai
from module import load_data, split_text_with_overlap, build_prompt
from dotenv import load_dotenv
import os
import time
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

# Load environment variables
load_dotenv()

# Initialize connections
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
llm = genai.GenerativeModel("gemini-2.0-flash")

# Load and prepare data
print("Loading documents...")
loaded_data = load_data(["PDF_Path1.pdf", "PDF_Path2.pdf", "so_on"]) #change the paths of pdf

# Split documents into chunks
all_chunks = []
for doc in loaded_data:
    all_chunks.extend(split_text_with_overlap(doc))
print(f"Split documents into {len(all_chunks)} chunks.")

# Load embedding model
print("Loading embedding model...")
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
dimension = embed_model.get_sentence_embedding_dimension()

# Setup Pinecone index
index_name = "my-docs-index"

print(f"Checking for existing index '{index_name}'...")
if index_name in pc.list_indexes().names():
    print("Index found. Deleting...")
    pc.delete_index(index_name)
    print("Waiting for deletion to complete...")
    time.sleep(10)  # Increased wait time

# Create new index
print("Creating new index...")
spec = ServerlessSpec(cloud="aws", region="us-east-1")
pc.create_index(
    name=index_name,
    dimension=dimension,
    metric="cosine",
    spec=spec
)

# Wait for index to be ready
print("Waiting for index to initialize...")
while not pc.describe_index(index_name).status['ready']:
    print("Index not ready yet, waiting...")
    time.sleep(10)

index = pc.Index(index_name)
print("Index is ready.")

# Additional wait time for serverless index to be fully operational
print("Waiting additional time for serverless index to be fully operational...")
time.sleep(60)  # Increased wait time for serverless

# Embed and store data
print("Embedding and upserting data to Pinecone...")
vectors_to_upsert = []
for idx, chunk in enumerate(all_chunks):
    embedding = embed_model.encode(chunk).tolist()
    vectors_to_upsert.append({
        "id": str(idx),
        "values": embedding,
        "metadata": {"text": chunk}
    })

# Upsert in smaller batches with delays
batch_size = 50
for i in range(0, len(vectors_to_upsert), batch_size):
    batch = vectors_to_upsert[i:i + batch_size]
    try:
        result = index.upsert(vectors=batch)
        print(f"Upsert result for batch {i//batch_size + 1}: {result}")
        time.sleep(2)  # Small delay between batches
    except Exception as e:
        print(f"Error upserting batch {i//batch_size + 1}: {e}")
        time.sleep(5)

print("Data upserted successfully.")

# Wait for upserts to be processed
print("Waiting for upserts to be processed...")
time.sleep(30)

# Check index stats multiple times until vectors appear
max_retries = 10
retry_count = 0
while retry_count < max_retries:
    stats = index.describe_index_stats()
    print(f"Index stats (attempt {retry_count + 1}): {stats}")
    
    if stats['total_vector_count'] > 0:
        print(f"Successfully stored {stats['total_vector_count']} vectors!")
        break
    else:
        print(f"No vectors found yet, waiting... (attempt {retry_count + 1}/{max_retries})")
        time.sleep(10)
        retry_count += 1

if stats['total_vector_count'] == 0:
    print("ERROR: No vectors were stored in the index!")
    exit(1)

# Query and generate response
question = input("enter your question")
print(f"\nQuestion: {question}")

# Create query embedding and search
query_embedding = embed_model.encode(question).tolist()
results = index.query(
    vector=query_embedding,
    top_k=5,
    include_metadata=True
)

print(f"Query results: {len(results['matches'])} matches found")

# Extract context from results
if results['matches']:
    retrieved_context = "\n".join([match['metadata']['text'] for match in results['matches']])
    print(f"\nRetrieved context preview: {retrieved_context[:200]}...")
    
    # Generate response
    final_prompt = build_prompt(retrieved_context, question)
    print("\nGenerating response...")
    response = llm.generate_content(final_prompt)
    print("\n--- Answer ---")
    print(response.text)
    print("--- End Answer ---")
else:
    print("ERROR: No matching documents found in the query results!")
    print("This suggests the vectors weren't properly stored or indexed.")