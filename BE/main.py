from fastapi import FastAPI, HTTPException, Body
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from sentence_transformers import SentenceTransformer
import json
import requests
from dotenv import load_dotenv
import os
import logging
import numpy as np
from pinecone import Pinecone, ServerlessSpec
from pinecone.grpc import PineconeGRPC as Pinecone
import pandas as pd

load_dotenv()

app = FastAPI()

# Add CORS middleware
origins = [
    "https://gelato-joy.vercel.app",  # Your frontend domain
    "http://localhost:3000",          # For local development (if needed)
    # Add more origins if needed
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load and parse the data
try:
    with open('ice_creams.json', 'r') as f:
        data = json.load(f)
except FileNotFoundError:
    logger.error("ice_creams.json file not found")
    exit(1)
except json.JSONDecodeError:
    logger.error("ice_creams.json file is malformed")
    exit(1)

# Extract the ice creams
ice_creams = data.get('ice_creams', [])

# Initialize the sentence transformer model
model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

# Initialize Pinecone
# Initialize Pinecone
pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))

index_name = os.getenv('PINECONE_INDEX_NAME')
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=1536,  # Make sure this matches the dimension of your embeddings
        metric="cosine", 
        spec=ServerlessSpec(
            cloud="aws", 
            region="us-east-1"
        ) 
    ) 

 # Use the existing index name

try:
    # Attempt to connect to the existing index
    index = pc.Index(index_name)
    
    # Get the index stats to determine the dimension
    index_stats = index.describe_index_stats()
    print(f"Total vector count: {index_stats.total_vector_count}")
    dimension = index_stats.dimension

    logger.info(f"Successfully connected to existing index: {index_name}")
    logger.info(f"Index dimension: {dimension}")
except Exception as e:
    logger.error(f"Error connecting to index {index_name}: {str(e)}")
    exit(1)

# Update the pad_vector function
def pad_vector(vector, target_dim=1536):
    if len(vector) >= target_dim:
        return vector[:target_dim]
    return np.pad(vector, (0, target_dim - len(vector)), 'constant')

# Create embeddings and upsert to Pinecone
# Create embeddings and upsert to Pinecone
passages = [f"{ice_cream['name']} - Flavors: {', '.join(ice_cream['flavors'])}. {ice_cream['review']}" for ice_cream in ice_creams]
embeddings = model.encode(passages)
vectors = [
    {
        "id": str(i),
        "values": pad_vector(embedding.tolist(), target_dim=dimension),
        "metadata": {"text": passages[i]}
    } for i, embedding in enumerate(embeddings)
]

df = pd.DataFrame(vectors)
index.upsert(vectors=vectors)

def generate_response(context: str, query: str) -> str:
    """
    Generate a response from the Hugging Face API
    """
    API_URL = "https://api-inference.huggingface.co/models/mistralai/Mixtral-8x7B-Instruct-v0.1"
    headers = {"Authorization": f"Bearer {os.getenv('HUGGINGFACE_API_KEY')}"}
    
    if not headers["Authorization"] or headers["Authorization"] == "Bearer None":
        logger.error("HUGGINGFACE_API_KEY environment variable is not set")
        return "I'm sorry, I'm not configured correctly. Please contact the administrator."
    
    prompt = f"""
    You are a friendly assistant for an ice cream shop called Gelato Joy. If the user's question is related to ice cream flavors or reviews, 
    use the following context to provide a detailed response. If the question is not related to ice creams 
    or if it's a general greeting, respond in a friendly manner and try to steer the conversation towards our ice cream offerings. Always maintain a cheerful and inviting tone.

    Context about ice creams: {context}

    Human: {query}

    Assistant: """
    
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_length": 150,
            "temperature": 0.7,
            "top_p": 0.9,
        }
    }
    
    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=10)
        response.raise_for_status()
        full_response = response.json()[0]['generated_text']
        
        # Extract only the Assistant's response
        assistant_response = full_response.split("Assistant:")[-1].strip()
        assistant_response = assistant_response.split("Human:")[0].strip()
        return assistant_response
    except requests.Timeout:
        logger.error("Request to Hugging Face API timed out")
        return "I'm sorry, it's taking longer than expected to respond. Could you please try again?"
    except requests.RequestException as e:
        logger.error(f"Error making request to Hugging Face API: {e}")
        logger.error(f"Response content: {e.response.content if e.response else 'No response'}")
        return "I'm having trouble connecting to my knowledge base right now. How about we chat about something else ice cream related?"
    except (KeyError, IndexError, json.JSONDecodeError) as e:
        logger.error(f"Error parsing response from Hugging Face API: {e}")
        logger.error(f"Response content: {response.content}")
        return "I understood your question about ice cream, but I'm having trouble formulating a response. Could you try rephrasing?"

@app.post("/query")
async def query(payload: dict = Body(...)):
    query = payload.get("query")
    if not query:
        raise HTTPException(status_code=400, detail="Query is required")

    ice_cream_keywords = ['ice cream', 'flavor', 'gelato', 'scoop', 'cone', 'sundae', 'nougat', 'escimo', 'coffee']
    ice_cream_names = [ice_cream['name'].lower() for ice_cream in ice_creams]
    
    is_ice_cream_related = any(keyword in query.lower() for keyword in ice_cream_keywords + ice_cream_names)

    if is_ice_cream_related:
        query_vector = pad_vector(model.encode([query])[0]).tolist()
        try:
            results = index.query(vector=query_vector, top_k=5, include_metadata=True)
            logger.info(f"Pinecone query results: {results}")
            context = " ".join([match.metadata['text'] for match in results.matches])
            
            # Find matching ice creams by name or flavor
            matching_ice_creams = []
            for ice_cream in ice_creams:
                if query.lower() in ice_cream['name'].lower() or \
                   any(query.lower() in flavor.lower() for flavor in ice_cream['flavors']):
                    matching_ice_creams.append(ice_cream)
            
            # Combine Pinecone results with name/flavor matches
            results = list({ice_cream['name']: ice_cream for ice_cream in matching_ice_creams + [ice_creams[int(match.id)] for match in results.matches]}.values())
        except Exception as e:
            logger.error(f"Error querying Pinecone: {str(e)}")
            context = "I'm having trouble accessing my ice cream database right now."
            results = []
    else:
        results = []
        context = "The user has asked a general question or greeted the assistant."

    ai_response = generate_response(context, query)
    
    logger.info(f"Query: {query}")
    logger.info(f"Context: {context}")
    logger.info(f"AI Response: {ai_response}")
    
    return JSONResponse(content={"results": results, "ai_response": ai_response})

@app.post("/add_ice_cream_wish")
async def add_ice_cream_wish(payload: dict = Body(...)):
    logger.info(f"Received payload: {payload}")
    
    name = payload.get("name")
    flavors = payload.get("flavors")
    description = payload.get("description")
    
    logger.info(f"Parsed data: name={name}, flavors={flavors}, description={description}")
    
    if not all([name, flavors, description]):
        logger.error("Missing required fields")
        raise HTTPException(status_code=400, detail="Name, flavors, and description are required")
    
    # Create a new ice cream entry
    new_ice_cream = {
        "name": name,
        "flavors": flavors.split(','),  # Assuming flavors are comma-separated
        "review": description,
        "stars": "New Wish"
    }
    
    # Create embedding
    passage = f"{new_ice_cream['name']} - Flavors: {', '.join(new_ice_cream['flavors'])}. {new_ice_cream['review']}"
    embedding = model.encode([passage])[0]
    
    # Prepare the vector for upserting
    vector = {
        "id": str(len(ice_creams)),
        "values": pad_vector(embedding.tolist(), target_dim=dimension),
        "metadata": {
            "name": name,
            "flavors": ', '.join(new_ice_cream['flavors']),
            "review": description,
            "stars": "New Wish"
        }
    }
    
    try:
        # Upsert the new ice cream wish
        index.upsert(vectors=[vector])
        
        # Add to the existing ice creams list
        ice_creams.append(new_ice_cream)

        # After successfully adding to Pinecone and the ice_creams list
        with open('ice_creams.json', 'w') as f:
            json.dump({"ice_creams": ice_creams}, f, indent=2)
        
        logger.info(f"Successfully upserted new ice cream wish")
        return JSONResponse(content={"message": "Ice cream wish added successfully"})
    except Exception as e:
        logger.error(f"Error during upsert operation: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))  # Default to 8000 if PORT is not set
    uvicorn.run(app, host="0.0.0.0", port=port)