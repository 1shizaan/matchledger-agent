import chromadb
import os # ADD THIS LINE
from openai import OpenAI # ADD THIS LINE
import numpy as np # ADD THIS LINE for cosine similarity if needed, though ChromaDB handles it internally

# �� Initialize OpenAI client globally within this file as well
try:
    openai_client = OpenAI() # Will pick up OPENAI_API_KEY from environment variables
    print("OpenAI client initialized for memory_db.")
except Exception as e:
    print(f"Error initializing OpenAI client in memory_db: {e}. Make sure OPENAI_API_KEY is set.")
    openai_client = None

# Remove the SentenceTransformer model loading:
# from sentence_transformers import SentenceTransformer
# model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize ChromaDB client and collection
client = chromadb.Client()
collection = client.get_or_create_collection(name="narration_pairs", metadata={"hnsw:space": "cosine"}) # Ensure cosine distance for consistency

def get_openai_embedding(text: str):
    """
    Generates an embedding for the given text using OpenAI's 'text-embedding-3-small' model.
    """
    if not openai_client:
        print("OpenAI client not initialized in memory_db. Cannot get embedding.")
        return None
    if not text.strip():
        # OpenAI API might return error for empty string, return a zero vector
        # Embedding size for text-embedding-3-small is 1536
        return [0.0] * 1536

    try:
        response = openai_client.embeddings.create(
            input=text,
            model="text-embedding-3-small" # Use the same model as recon_engine.py
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Error getting embedding for text '{text}' in memory_db: {e}")
        return None

def add_to_memory(ledger_text, bank_text):
    # Combine texts for the embedding to represent the pair
    combined_text = f"{ledger_text} | {bank_text}"
    query_id = f"{ledger_text}-{bank_text}".replace(" ", "_").replace("|", "_") # Clean ID for ChromaDB

    embedding = get_openai_embedding(combined_text)

    if embedding is not None:
        try:
            collection.add(
                documents=[combined_text],
                ids=[query_id],
                embeddings=[embedding]
            )
            print(f"Added '{combined_text}' to memory DB.")
        except Exception as e:
            print(f"Error adding to ChromaDB: {e}")
    else:
        print(f"Skipped adding '{combined_text}' to memory due to embedding failure.")


def is_match_from_memory(ledger_text, bank_text, threshold=0.75): # Adjust threshold as needed for OpenAI
    """
    Checks if a match for the narration pair exists in memory using OpenAI embeddings.
    For cosine distance in ChromaDB, a lower distance means higher similarity.
    Distance = 1 - Similarity. So for similarity >= threshold, distance <= (1 - threshold).
    """
    query_text = f"{ledger_text} | {bank_text}"
    query_embedding = get_openai_embedding(query_text)

    if query_embedding is None:
        return False

    try:
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=1,
            include=['distances']
        )

        # Check if any results were found and if the distance meets the criteria
        if results and results["distances"] and len(results["distances"][0]) > 0:
            distance = results["distances"][0][0]
            # Print for debugging
            print(f"Memory DB query for '{query_text}': Distance {distance:.2f}, Threshold for distance: {(1 - threshold):.2f}")
            # If distance is less than or equal to (1 - threshold), it's a match
            return distance <= (1 - threshold)
    except Exception as e:
        print(f"Error querying ChromaDB: {e}")
        return False

    return False