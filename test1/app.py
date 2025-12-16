import streamlit as st
import torch
from PIL import Image
from torchvision import transforms
from model import load_model

st.title("Pneumonia Detection")

model = load_model()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

uploaded = st.file_uploader("Upload Chest X-ray", type=["jpg", "png", "jpeg"])

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Uploaded Image")

    img = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(img)
        pred = torch.argmax(output, dim=1).item()

    label = "PNEUMONIA" if pred == 1 else "NORMAL"
    st.success(f"Prediction: {label}")
