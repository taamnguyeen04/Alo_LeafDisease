import torch
import torch.nn as nn
from torchvision.models import vit_b_16, ViT_B_16_Weights
from PIL import Image
import numpy as np
import os

# ================================
# SIMPLE BG REMOVAL (PIL + NumPy)
# ================================
def remove_bg_pil(img):
    """
    Remove background bằng ngưỡng màu xanh.
    Lá cây thường có màu xanh > đỏ và xanh > xanh dương.
    """

    img_np = np.array(img)
    r = img_np[:, :, 0]
    g = img_np[:, :, 1]
    b = img_np[:, :, 2]

    # Mask: chọn pixel có màu xanh vượt trội
    mask = (g > r + 20) & (g > b + 20)

    # Apply mask
    img_np[~mask] = [0, 0, 0]  # nền đen

    return Image.fromarray(img_np)


# ================================
# LOAD MODEL
# ================================
def load_model(checkpoint_path, num_classes=8):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    weights = ViT_B_16_Weights.IMAGENET1K_V1
    model = vit_b_16(weights=weights)
    model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)

    checkpoint = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model, weights.transforms(), device


# ================================
# INFERENCE FUNCTION
# ================================
def infer(image_path, checkpoint="./checkpoints_stage1/best.pt"):
    if not os.path.exists(image_path):
        raise FileNotFoundError("Ảnh không tồn tại!")

    # Load model + transform
    model, transform, device = load_model(checkpoint)

    # Load ảnh gốc
    img = Image.open(image_path).convert("RGB")

    # ===== REMOVE BACKGROUND =====
    img = remove_bg_pil(img)
    # =============================

    # Transform
    img_t = transform(img).unsqueeze(0).to(device)

    # Predict
    with torch.no_grad():
        output = model(img_t)
        probs = torch.softmax(output, dim=1)[0]
        pred_cls = torch.argmax(probs).item()

    # Class names
    class_dir = "C:/Users/tam/Desktop/Data/leaf/PlantVillage_Stage1_PlantType/test"
    class_names = sorted(os.listdir(class_dir))

    print("\n====== RESULT ======")
    print(f"Predicted class: {class_names[pred_cls]}")
    print(f"Confidence: {probs[pred_cls].item():.4f}")
    print("====================\n")

    return class_names[pred_cls], probs[pred_cls].item()


# ================================
# RUN EXAMPLE
# ================================
if __name__ == "__main__":
    image_path = "C:/Users/tam/Downloads/benh-phan-trang-gay-hai-tao.jpg"
    infer(image_path)
