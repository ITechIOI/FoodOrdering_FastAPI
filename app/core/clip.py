import clip
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

def load_clip_model():
    model, preprocess = clip.load("ViT-B/32", device=device)
    model.eval()
    return model, preprocess

model, preprocess = load_clip_model()
