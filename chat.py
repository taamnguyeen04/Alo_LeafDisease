import os
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import vit_b_16, ViT_B_16_Weights
from PIL import Image
import numpy as np

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv

# ================== PHẦN 1: LOAD MODEL VIỆT ==================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model đã train
checkpoint = torch.load("./checkpoints/best.pt", map_location=device)
weights = ViT_B_16_Weights.IMAGENET1K_V1
model = vit_b_16(weights=weights)
num_classes = len(os.listdir(r"C:\Users\tam\Documents\data\PlantVillage_Split\test"))
model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
model.load_state_dict(checkpoint["model_state_dict"])
model.to(device)
model.eval()

transform = weights.transforms()
class_names = os.listdir(r"C:\Users\tam\Documents\data\PlantVillage_Split\test")

def predict_disease(image_path):
    img = Image.open(image_path).convert("RGB")
    img_t = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(img_t)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
        pred_idx = np.argmax(probs)
    return class_names[pred_idx], probs[pred_idx]


# ================== PHẦN 2: TẠO CHATBOT NÔNG NGHIỆP ==================
load_dotenv()

# Load tài liệu nông nghiệp (text file)
loader = TextLoader("books/huong_dan_trong_cay.txt", encoding="utf-8")   # thay bằng sách thật
documents = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
chunks = splitter.split_documents(documents)
combined_text = "\n\n".join([chunk.page_content for chunk in chunks])  # lấy hết nội dung

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.3,
    google_api_key=os.getenv("GOOGLE_API_KEY")
)


def chatbot(query, context_info=""):
    messages = [
        SystemMessage(content="Bạn là trợ lý AI chuyên về nông nghiệp. Trả lời thân thiện, dễ hiểu cho nông dân."),
        HumanMessage(content=f"Tài liệu:\n{combined_text}\n\nThông tin bệnh cây: {context_info}\n\nCâu hỏi: {query}")
    ]
    response = llm.invoke(messages)
    return response.content


# ================== PHẦN 3: DEMO ==================
if __name__ == "__main__":
    print("🌿 Nhập đường dẫn ảnh lá cây (hoặc gõ 'skip' để bỏ qua nhận diện).")
    img_path = input("Ảnh lá cây: ").strip().strip('"')

    disease_info = ""
    if img_path.lower() != "skip":
        disease, confidence = predict_disease(img_path)
        disease_info = f"Hệ thống dự đoán lá cây bị: {disease} (độ tin cậy {confidence:.2f})"
        print("🔍", disease_info)

    print("\n🤖 Chatbot nông nghiệp sẵn sàng! Gõ 'exit' để thoát.")
    while True:
        query = input("👤 Bạn: ")
        if query.lower() in ["exit", "quit"]:
            break

        answer = chatbot(query, context_info=disease_info)
        print("\n🤖 Bot:", answer)
