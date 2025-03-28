import os
from pinecone import Pinecone
from app.core.config import settings

pc = Pinecone(api_key=settings.PINECONE_API_KEY)

index = pc.Index(settings.PINECONE_INDEX)

def query_pinecone(vector, top_k=5):
    result = index.query(vector=vector.tolist(), top_k=top_k, include_metadata=True)
    return result


