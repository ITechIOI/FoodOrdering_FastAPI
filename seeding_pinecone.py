import requests
from PIL import Image
import io
import torch
import clip
from pinecone import Pinecone
from app.core.config import settings

# Load CLIP
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
model.eval()

# Kết nối Pinecone
pc = Pinecone(api_key=settings.PINECONE_API_KEY)
index = pc.Index(settings.PINECONE_INDEX)

# Gọi dữ liệu món ăn từ NestJS GraphQL
def get_menu_items():
    query = """
    {
        menus {
            id
            name
            description
            imageUrl
        }
    }
    """
    response = requests.post(
        settings.NESTJS_MENU_ENDPOINT,
        json={"query": query}
    )
    response.raise_for_status()
    data = response.json()
    return data["data"]["menus"]

# Hàm sinh vector embedding từ ảnh
@torch.no_grad()
def encode_image_from_url(image_url):
    image = Image.open(io.BytesIO(requests.get(image_url).content)).convert("RGB")
    image_input = preprocess(image).unsqueeze(0).to(device)
    image_features = model.encode_image(image_input)
    image_features /= image_features.norm(dim=-1, keepdim=True)
    return image_features.cpu().numpy()[0]

# Insert tất cả dữ liệu vào Pinecone
def seed():
    menu_items = get_menu_items()
    vectors = []

    for item in menu_items:
        try:
            vector = encode_image_from_url(item["imageUrl"])
            vectors.append({
                "id": str(item["id"]),
                "values": vector.tolist(),
                "metadata": {
                    "label": item["name"],
                    "description": item["description"]
                }
            })
            print(f"✅ Added vector for {item['name']}")
        except Exception as e:
            print(f"❌ Failed for {item['name']}: {e}")

    # Gửi lên Pinecone
    if vectors:
        index.upsert(vectors=vectors)
        print(f"\n🌟 Đã upsert {len(vectors)} vectors vào Pinecone thành công.")
    else:
        print("⚠️ Không có vector nào được chèn.")

if __name__ == "__main__":
    seed()
