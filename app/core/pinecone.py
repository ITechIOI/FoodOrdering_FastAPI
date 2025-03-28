from pinecone import Pinecone
from dotenv import load_dotenv
import os

load_dotenv()
API_KEY = os.getenv("PINECONE_API_KEY")

pc = Pinecone(API_KEY)