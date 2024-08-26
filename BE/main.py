from fastapi import FastAPI, HTTPException, Body, Request
from fastapi.responses import JSONResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from sentence_transformers import SentenceTransformer
import json
import requests
from dotenv import load_dotenv
import os
import logging
import numpy as np
from pinecone import Pinecone, ServerlessSpec
import pandas as pd

# Load environment variables
load_dotenv()

PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_INDEX_NAME = os.getenv('PINECONE_INDEX_NAME')
HUGGINGFACE_API_KEY = os.getenv('HUGGINGFACE_API_KEY')

if not all([PINECONE_API_KEY, PINECONE_INDEX_NAME, HUGGINGFACE_API_KEY]):
    logging.error("Missing required environment variables")
    exit(1)

app = FastAPI()

# CORS middleware
origins = [
    "https://gelato-joy.vercel.app",  
    "http://localhost:3000",
    "https://gelato-joy-t765.onrender.com"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_ice_cream_data():
    try:
        with open('ice_creams.json', 'r') as f:
            return json.load(f).get('ice_creams', [])
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.error(f"Error loading ice_creams.json: {e}")
        exit(1)

ice_creams = load_ice_cream_data()

model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

def create_pinecone_index():
    pc = Pinecone(api_key=PINECONE_API_KEY)
    if PINECONE_INDEX_NAME not in pc.list_indexes().names():
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=1536,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
    return pc.Index(PINECONE_INDEX_NAME)

index = create_pinecone_index()

def pad_vector(vector, target_dim=1536):
    return np.pad(vector, (0, max(0, target_dim - len(vector))), 'constant')

passages = [f"{ice_cream['name']} - Flavors: {', '.join(ice_cream['flavors'])}. {ice_cream['review']}" for ice_cream in ice_creams]
embeddings = model.encode(passages)
vectors = [
    {
        "id": str(i),
        "values": pad_vector(embedding.tolist()),
        "metadata": {"text": passages[i]}
    } for i, embedding in enumerate(embeddings)
]

df = pd.DataFrame(vectors)
index.upsert(vectors=vectors)

def call_huggingface_api(prompt):
    API_URL = "https://api-inference.huggingface.co/models/mistralai/Mixtral-8x7B-Instruct-v0.1"
    headers = {"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"}
    payload = {
        "inputs": prompt,
        "parameters": {"max_length": 500, "temperature": 0.7, "top_p": 0.9}
    }
    response = requests.post(API_URL, headers=headers, json=payload, timeout=10)
    response.raise_for_status()
    return response.json()[0]['generated_text']

def generate_response(context: str, query: str) -> str:
    prompt = f"""
    You are a friendly assistant for an ice cream shop called Gelato Joy. If the user's question is related to ice cream flavors or reviews, 
    use the following context to provide a detailed response. If the question is not related to ice creams 
    or if it's a general greeting, respond in a friendly manner and try to steer the conversation towards our ice cream offerings, though you can talk about everything. Always maintain a cheerful and inviting tone.

    Context about ice creams: {context}

    Human: {query}

    Assistant: """
    
    try:
        full_response = call_huggingface_api(prompt)
        assistant_response = full_response.split("Assistant:")[-1].strip().split("Human:")[0].strip()
        return assistant_response or "I seem to have missed part of my response. Let me try again!"
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        return "I'm having trouble connecting to my knowledge base right now. How about we chat about something else ice cream related?"

app.route("/health", methods=["GET", "HEAD"])
async def health_check(request: Request):
    try:
        if request.method == "HEAD":
            return Response(status_code=200)
        
        return JSONResponse(content={"status": "ok"})
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return Response(status_code=500)

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
            
            matching_ice_creams = []
            for ice_cream in ice_creams:
                if query.lower() in ice_cream['name'].lower() or \
                   any(query.lower() in flavor.lower() for flavor in ice_cream['flavors']):
                    matching_ice_creams.append(ice_cream)
            
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
    name = payload.get("name")
    flavors = payload.get("flavors")
    description = payload.get("description")
    
    if not all([name, flavors, description]):
        logger.error("Missing required fields")
        raise HTTPException(status_code=400, detail="Name, flavors, and description are required")
    
    new_ice_cream = {
        "name": name,
        "flavors": flavors.split(','), 
        "review": description,
        "stars": "New Wish"
    }
    
    passage = f"{new_ice_cream['name']} - Flavors: {', '.join(new_ice_cream['flavors'])}. {new_ice_cream['review']}"
    embedding = model.encode([passage])[0]
    
    vector = {
        "id": str(len(ice_creams)),
        "values": pad_vector(embedding.tolist()),
        "metadata": {
            "name": name,
            "flavors": ', '.join(new_ice_cream['flavors']),
            "review": description,
            "stars": "New Wish"
        }
    }
    
    try:
        index.upsert(vectors=[vector])
        
        ice_creams.append(new_ice_cream)

        with open('ice_creams.json', 'w') as f:
            json.dump({"ice_creams": ice_creams}, f, indent=2)
        
        logger.info(f"Successfully upserted new ice cream wish")
        return JSONResponse(content={"message": "Ice cream wish added successfully"})
    except Exception as e:
        logger.error(f"Error during upsert operation: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))  
    uvicorn.run(app, host="0.0.0.0", port=port)