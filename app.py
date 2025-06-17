import streamlit as st
import torch
import tempfile
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models
import imageio
import os

# Define extract_frames and predict_clip here or import from utils
def extract_frames(video_path, frame_rate=10):
    """Extract frames from video at specified frame rate using imageio."""
    reader = imageio.get_reader(video_path, "ffmpeg")
    frames = []
    for i, frame in enumerate(reader):
        if i % frame_rate == 0:
            pil_img = Image.fromarray(frame)
            frames.append(pil_img)
    reader.close()
    return frames


def predict_clip(frames, model, transform, device='cpu'):
    """Predict class of clip based on average of frame predictions."""
    model.eval()
    preds = []
    with torch.no_grad():
        for frame in frames:
            img = transform(frame).unsqueeze(0).to(device)
            output = model(img)
            pred = output.argmax(dim=1, keepdim=True)
            preds.append(pred.item())
    # Majority vote
    final_pred = max(set(preds), key=preds.count)
    return final_pred


# Load model architecture and weights
@st.cache_resource
def load_model():
    model = models.resnet18(pretrained=False)
    num_features = model.fc.in_features
    model.fc = torch.nn.Linear(num_features, 2)  # 2 classes: Accident / Non-Accident
    model.load_state_dict(torch.load('accident_detection_resnet.pth', map_location='cpu'))
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    return model, device


# Transform pipeline
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# Streamlit UI
st.title("🚗 Accident Detection in Video")
uploaded_video = st.file_uploader("Upload a video clip", type=["mp4", "avi", "mov"])

if uploaded_video is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(uploaded_video.read())
    video_path = tfile.name
    tfile.close()

    st.video(video_path)
    st.write("⏳ Processing...")

    model, device = load_model()
    frames = extract_frames(video_path, frame_rate=10)

    predicted_index = predict_clip(frames, model, transform, device)

    label_map = {0: 'Accident', 1: 'Non-Accident'}
    prediction_label = label_map[predicted_index]

    st.success(f"✅ Prediction: **{prediction_label}**")

    # Clean up temp file
    os.unlink(video_path)
