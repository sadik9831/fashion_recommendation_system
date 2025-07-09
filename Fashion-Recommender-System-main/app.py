import streamlit as st
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50
from torch import nn
import numpy as np
import os
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity

# Load image paths
IMAGE_FOLDER = "reduced_dataset"
IMAGE_PATHS = [os.path.join(IMAGE_FOLDER, fname) for fname in os.listdir(IMAGE_FOLDER) if fname.lower().endswith(('.jpg', '.jpeg', '.png'))]

# Load and freeze ResNet50 for feature extraction
@st.cache_resource
def get_model():
    model = resnet50(pretrained=True)
    model.fc = nn.Identity()  # remove classification layer
    model.eval()
    return model

resnet = get_model()

# Image transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Function to extract features from image
def extract_features(image_input):
    if isinstance(image_input, str):  # path
        img = Image.open(image_input).convert('RGB')
    else:  # uploaded file
        img = Image.open(image_input).convert('RGB')

    img_t = transform(img).unsqueeze(0)
    with torch.no_grad():
        features = resnet(img_t)
    return features.squeeze(0).numpy()

# Extract features for all dataset images
@st.cache_data
def load_dataset_features(_model):  # Added underscore to prevent hashing
    features = []
    for path in IMAGE_PATHS:
        try:
            feat = extract_features(path)
            features.append(feat)
        except Exception as e:
            st.warning(f"Error processing {path}: {e}")
            continue
    return np.array(features)

# Load dataset features with error handling
try:
    dataset_features = load_dataset_features(resnet)
    if len(dataset_features) == 0:
        st.error("No valid images found in the dataset.")
        st.stop()
except Exception as e:
    st.error(f"Failed to load dataset features: {e}")
    st.stop()

# UI
st.title("🧥 Fashion Recommender System")
st.write("Upload a fashion image to find similar styles")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    try:
        input_img = Image.open(uploaded_file).convert("RGB")
        st.image(input_img, caption="Uploaded Image", use_container_width=True)

        # Extract feature from uploaded image
        input_feature = extract_features(uploaded_file)
        input_feature = input_feature.reshape(1, -1)

        # Compute similarity
        sims = cosine_similarity(input_feature, dataset_features)[0]
        top5_idx = np.argsort(sims)[-5:][::-1]
        top5_scores = sims[top5_idx]

        st.subheader("Top 5 Similar Items")
        cols = st.columns(5)
        for idx, score, col in zip(top5_idx, top5_scores, cols):
            try:
                col.image(IMAGE_PATHS[idx], use_container_width=True)
                col.write(f"Similarity: {score:.2%}")
            except Exception as e:
                col.error(f"Error displaying image: {e}")

        # Display average similarity score
        avg_similarity = np.mean(top5_scores)
        st.info(f"Average similarity score for top 5 recommendations: {avg_similarity:.2%}")

    except Exception as e:
        st.error(f"An error occurred during processing: {e}")
