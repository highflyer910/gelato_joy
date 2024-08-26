from fastapi import FastAPI, HTTPException, Body
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from sentence_transformers import SentenceTransformer
import json, os, logging, requests, numpy as np
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

app = FastAPI()

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://gelato-joy.vercel.app",  
        "http://localhost:3000",
        "https://gelato-joy-t765.onrender.com"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load ice cream data
try:
    with open('ice_creams.json', 'r') as f:
        ice_creams = json.load(f).get('ice_creams', [])
except (FileNotFoundError, json.JSONDecodeError) as e:
    logger.error(f"Error loading ice cream data: {e}")
    exit(1)

model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

# Pinecone setup
pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
index_name = os.getenv('PINECONE_INDEX_NAME')

if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

index = pc.Index(index_name)

# embeddings and upsert vectors
passages = [f"{ic['name']} - Flavors: {', '.join(ic['flavors'])}. {ic['review']}" for ic in ice_creams]
vectors = [{
    "id": str(i),
    "values": np.pad(model.encode([passages[i]])[0], (0, 1536 - len(model.encode([passages[i]])[0])), 'constant').tolist(),
    "metadata": {"text": passages[i]}
} for i in range(len(passages))]

index.upsert(vectors=vectors)

# Generate AI response
def generate_response(context: str, query: str) -> str:
    API_URL = "https://api-inference.huggingface.co/models/mistralai/Mixtral-8x7B-Instruct-v0.1"
    headers = {"Authorization": f"Bearer {os.getenv('HUGGINGFACE_API_KEY')}"}
    
    prompt = f"""
    You are a friendly assistant for an ice cream shop called Gelato Joy. If the user's question is related to ice cream flavors or reviews, 
    use the following context to provide a detailed response. If the question is not related to ice creams 
    or if it's a general greeting, respond in a friendly manner and try to steer the conversation towards our ice cream offerings, though you can talk about everything. Always maintain a cheerful and inviting tone.

    Context about ice creams: {context}

    Human: {query}

    Assistant: """
    
    try:
        response = requests.post(API_URL, headers=headers, json={"inputs": prompt, "parameters": {"max_length": 150, "temperature": 0.7, "top_p": 0.9}})
        return response.json()[0]['generated_text'].split("Assistant:")[-1].strip().split("Human:")[0].strip()
    except requests.RequestException as e:
        logger.error(f"Error: {e}")
        return "I'm having trouble connecting to my knowledge base right now. How about we chat about something else ice cream related?"

@app.post("/query")
async def query(payload: dict = Body(...)):
    query = payload.get("query")
    if not query:
        raise HTTPException(status_code=400, detail="Query is required")
    
    is_ice_cream_related = any(keyword in query.lower() for keyword in ['ice cream', 'flavor', 'gelato', 'scoop', 'cone', 'sundae', 'nougat', 'escimo', 'coffee']) or \
                           any(ic['name'].lower() in query.lower() for ic in ice_creams)

    if is_ice_cream_related:
        query_vector = np.pad(model.encode([query])[0], (0, 1536 - len(model.encode([query])[0])), 'constant').tolist()
        results = index.query(vector=query_vector, top_k=5, include_metadata=True).matches
        context = " ".join([match.metadata['text'] for match in results])
        matching_ice_creams = [ic for ic in ice_creams if any(q in ic['name'].lower() or q in flavor.lower() for flavor in ic['flavors'])]
        results = list({ic['name']: ic for ic in matching_ice_creams + [ice_creams[int(match.id)] for match in results]}.values())
    else:
        context = "The user has asked a general question or greeted the assistant."
        results = []

    ai_response = generate_response(context, query)
    
    return JSONResponse(content={"results": results, "ai_response": ai_response})

@app.post("/add_ice_cream_wish")
async def add_ice_cream_wish(payload: dict = Body(...)):
    name, flavors, description = payload.get("name"), payload.get("flavors"), payload.get("description")
    if not all([name, flavors, description]):
        raise HTTPException(status_code=400, detail="Name, flavors, and description are required")

    new_ice_cream = {"name": name, "flavors": flavors.split(','), "review": description, "stars": "New Wish"}
    passage = f"{new_ice_cream['name']} - Flavors: {', '.join(new_ice_cream['flavors'])}. {new_ice_cream['review']}"
    embedding = model.encode([passage])[0]
    vector = {"id": str(len(ice_creams)), "values": np.pad(embedding, (0, 1536 - len(embedding)), 'constant').tolist(), "metadata": new_ice_cream}

    try:
        index.upsert(vectors=[vector])
        ice_creams.append(new_ice_cream)
        with open('ice_creams.json', 'w') as f:
            json.dump({"ice_creams": ice_creams}, f, indent=2)
        return JSONResponse(content={"message": "Ice cream wish added successfully"})
    except Exception as e:
        logger.error(f"Error during upsert operation: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))  
    uvicorn.run(app, host="0.0.0.0", port=port)
