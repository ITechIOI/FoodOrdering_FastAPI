from fastapi import FastAPI
from app.api.menu import router as menu_router
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(menu_router)

@app.get("/search-similar")
def search_similar_food(image_path, top_k=5):
    return {"image_path": image_path, "top_k": top_k}