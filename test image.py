import os
import torch
import torch.nn as nn
from torchvision.models import vit_b_16, ViT_B_16_Weights
from torchvision import transforms
from PIL import Image

# ==== CONFIG ====
checkpoint_path = "./checkpoints/best.pt"   # model đã lưu
img_path = r"C:\Users\tam\Documents\data\PlantVillage_Split\test\mango_anthracnose\20211011_133426 (Custom).jpg"  # ảnh cần test
class_dir = r"C:\Users\tam\Documents\data\PlantVillage_Split\test"  # để lấy tên class

# ==== LOAD MODEL ====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
weights = ViT_B_16_Weights.IMAGENET1K_V1
model = vit_b_16(weights=weights)

num_classes = len(os.listdir(class_dir))
model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)

checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
model.to(device)
model.eval()

# ==== TRANSFORM ====
transform = weights.transforms()

# ==== LOAD IMAGE ====
img = Image.open(img_path).convert("RGB")
img_tensor = transform(img).unsqueeze(0).to(device)  # thêm batch dim

# ==== PREDICT ====
with torch.no_grad():
    outputs = model(img_tensor)
    probs = torch.softmax(outputs, dim=1)
    pred_class = torch.argmax(probs, dim=1).item()

# ==== CLASS NAME ====
categories = sorted(os.listdir(class_dir))
print("Predicted:", categories[pred_class])
print("Confidence:", probs[0][pred_class].item())
