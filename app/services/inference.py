from PIL import Image
import io
import torch
from app.core.clip import model, preprocess, device

@torch.no_grad()
def image_to_embedding(image_bytes: bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image_input = preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image_input)
        image_features /= image_features.norm(dim=-1, keepdim=True)

    return image_features.cpu().numpy()[0]
