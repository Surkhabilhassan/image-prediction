import io
import json
import torch
import streamlit as st
from PIL import Image
from torchvision import transforms, models
from pathlib import Path
css_path = Path("style.css")
if css_path.exists():
    st.markdown(f"<style>{css_path.read_text()}</style>", unsafe_allow_html=True)


st.set_page_config(page_title="Image Classifier", layout="centered")
st.title("Image Classifier — ‘What is this image of?’")
st.write("Upload an image, and the model will tell you what it is. If a custom model (model.pth) is found, it will be used; otherwise, a pretrained ImageNet model will be used.")


@st.cache_resource
def load_model():
    ckpt_path = Path("model.pth")
    mapping_path = Path("class_to_idx.json")

    if ckpt_path.exists() and mapping_path.exists():
       
        num_classes = None
        with open(mapping_path, "r", encoding="utf-8") as f:
            class_to_idx = json.load(f)
        idx_to_class = {v: k for k, v in class_to_idx.items()}
        num_classes = len(idx_to_class)

        model = models.resnet18(weights=None)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
        model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
        model.eval()

        class_names = [idx_to_class[i] for i in range(num_classes)]
        source = "custom"
    else:
        
        weights = models.ResNet18_Weights.IMAGENET1K_V1
        model = models.resnet18(weights=weights)
        model.eval()
        class_names = weights.meta["categories"]
        source = "imagenet"

    return model, class_names, source

@st.cache_resource
def get_transform():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

def predict(img_pil, model, class_names, topk=5):
    tfm = get_transform()
    with torch.no_grad():
        x = tfm(img_pil).unsqueeze(0)  
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]
        top_probs, top_idxs = torch.topk(probs, k=min(topk, len(class_names)))
        results = [(class_names[i], float(top_probs[j])) for j, i in enumerate(top_idxs.tolist())]
    return results


st.sidebar.header("Model Info")
model, class_names, source = load_model()
st.sidebar.write(f"**Model source:** `{source}`")
st.sidebar.write(f"**Classes:** {len(class_names)}")


uploaded = st.file_uploader("Upload an image here (JPG/PNG)", type=["jpg","jpeg","png"])
camera = st.camera_input("Or take a photo directly from the camera (optional)")

img_bytes = None
if uploaded is not None:
    img_bytes = uploaded.read()
elif camera is not None:
    img_bytes = camera.read()

if img_bytes:
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Thinking…"):
        results = predict(img, model, class_names, topk=5)

    st.subheader("🔎 Prediction")
    top_label, top_score = results[0]
    st.markdown(f"**My strongest guess:** `{top_label}`  (confidence: {top_score:.2%})")

    st.write("**Top-5 guesses:**")
    for lbl, p in results:
        st.write(f"- {lbl} — {p:.2%}")
else:
    st.warning("Upload an image or take a photo with your camera, then see the prediction.")
