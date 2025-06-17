import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def extract_frames(video_path, frame_rate=1):
    frames = []
    cap = cv2.VideoCapture(video_path)
    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if count % frame_rate == 0:
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            frames.append(img)
        count += 1
    cap.release()
    return frames

def predict_clip(frames, model, transform, threshold=0.5):
    model.eval()
    accident_votes = 0
    with torch.no_grad():
        for frame in frames:
            input_tensor = transform(frame).unsqueeze(0).to(device)
            output = model(input_tensor)
            pred = torch.softmax(output, dim=1)
            accident_votes += pred.argmax().item()
    avg_vote = accident_votes / len(frames)
    return "Accident" if avg_vote > threshold else "Non-Accident"
