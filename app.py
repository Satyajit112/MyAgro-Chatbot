import streamlit as st
import json
import cv2
from PIL import Image
from utils import load_and_preprocess_image, is_plant_image, load_model
import torch
import torch.nn.functional as F
import os
from datetime import datetime
import requests
from io import BytesIO
import numpy as np  

# Page configuration
st.set_page_config(
    page_title="Plant Disease Classifier",
    page_icon="🌿",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        padding: 0.5rem;
        border-radius: 5px;
        border: none;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .title-container {
        background-color: #f0f8f0;
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
    }
    .project-info {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .upload-section {
        background-color: #ffffff;
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .prediction-section {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        margin-top: 1rem;
    }
    .confidence-bar {
        padding: 0.5rem;
        margin: 0.5rem 0;
        border-radius: 5px;
    }
    .footer {
        text-align: center;
        padding: 1rem;
        background-color: #f0f8f0;
        border-radius: 10px;
        margin-top: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

# Model loading
model_path = './model.pth'
with open("class_indices.json", "r") as f:
    class_indices = json.load(f)
model = load_model(model_path)

def get_display_name(class_name):
    """Convert class name to display friendly format"""
    return class_name.replace('_', ' ').title()

def predict_image_class(image):
    with torch.no_grad():
        img_tensor = load_and_preprocess_image(image)
        outputs = model(img_tensor)
        probabilities = F.softmax(outputs, dim=1)
        
        # Get top 3 predictions
        top_probs, top_indices = torch.topk(probabilities, 3)
        
        predictions = []
        for i in range(3):
            idx = top_indices[0][i].item()
            prob = top_probs[0][i].item() * 100
            label = class_indices.get(str(idx), "Unknown")
            predictions.append((label, prob))
        
        return predictions
def get_disease_from_image_url(image_url):
    """
    Processes an image from a URL and returns the detected disease or a message.

    Parameters:
        image_url (str): The URL of the image.

    Returns:
        str: The disease name or a message indicating the image is not a plant.
    """
    try:
        # Download the image
        response = requests.get(image_url)
        response.raise_for_status()  # Ensure the request was successful
        image = Image.open(BytesIO(response.content))

        # Check if it's a plant image
        if not is_plant_image(image):
            return "The provided image is not of a plant."

        # Preprocess the image
        image_tensor = load_and_preprocess_image(image)

        # Perform prediction
        model.eval()
        with torch.no_grad():
            outputs = model(image_tensor)
            _, predicted = torch.max(outputs, 1)
            class_id = predicted.item()
            class_name = list(class_indices.values())[class_id]
        
        return f"The detected plant disease is: {class_name}"
    except Exception as e:
        return f"Error processing the image: {str(e)}"
    
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    stframe = st.empty()
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if not is_plant_image(img_pil):
            cv2.putText(frame, "Plant or leaf not detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        else:
            predictions = predict_image_class(img_pil)
            top_prediction = predictions[0]
            cv2.putText(frame, f"Prediction: {top_prediction[0]} ({top_prediction[1]:.1f}%)", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        
        stframe.image(frame, channels="BGR", use_column_width=True)
    cap.release()

# Header Section
st.markdown('<div class="title-container">', unsafe_allow_html=True)
st.title("🌿 Plant Disease Classifier")
st.markdown("Using VGG-11 Architecture")
st.markdown('</div>', unsafe_allow_html=True)

# Project Information
st.markdown('<div class="project-info">', unsafe_allow_html=True)
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### 👥 Group Details")
    st.write("**Group Name:** AI-C-4")
    st.write("**Guide Name:** PROF. ZARINABEGAM MUNDARGI")
    
with col2:
    st.markdown("### 🏛️ Institution")
    st.write("**College:** Vishwakarma Institute of Technology,Pune")
    st.write("**Department:** Artificial intelligence and Data science")
    
with col3:
    st.markdown("### 📊 Project Stats")
    st.write("**Model:** VGG-11")
    st.write("**Training Accuracy:** 99.39%")
    st.write("**Validation Accuracy:** 97.58%")
    st.write("**Overall Accuracy:** 97.58%")
st.markdown('</div>', unsafe_allow_html=True)

# Main Interface
st.markdown('<div class="upload-section">', unsafe_allow_html=True)
file_type = st.selectbox("📁 Choose Input Type", ["Image", "Video", "Image URL"])

if file_type == "Image":
    uploaded_image = st.file_uploader("Upload a plant image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_image is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📸 Uploaded Image")
            image = Image.open(uploaded_image).convert("RGB")
            st.image(image, use_column_width=True)
        
        with col2:
            st.markdown("### 🔍 Analysis")
            if not is_plant_image(image):
                st.error("⚠️ Invalid image: not enough green content to classify as a plant.")
            else:
                if st.button("🔍 Analyze Image"):
                    with st.spinner("Analyzing image..."):
                        predictions = predict_image_class(image)
                        
                        st.markdown('<div class="prediction-section">', unsafe_allow_html=True)
                        st.markdown("### 🎯 Results")
                        
                        # Display top 3 predictions with confidence bars
                        for i, (label, confidence) in enumerate(predictions):
                            display_name = get_display_name(label)
                            color = "#4CAF50" if i == 0 else "#90EE90"
                            st.markdown(f"**{i+1}. {display_name}**")
                            st.markdown(
                                f"""
                                <div class="confidence-bar" 
                                     style="background-color: {color}; 
                                            width: {confidence}%">
                                    {confidence:.1f}%
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Additional information for top prediction
                        if predictions:
                            st.markdown("### 📋 Detailed Analysis")
                            st.info(
                                f"""
                                The model is most confident that this is a case of **{get_display_name(predictions[0][0])}** 
                                with a confidence of {predictions[0][1]:.1f}%.
                                """
                            )

elif file_type == "Video":
    uploaded_video = st.file_uploader("Upload a video...", type=["mp4"])
    
    if uploaded_video is not None:
        # Create temp directory if it doesn't exist
        os.makedirs("temp", exist_ok=True)
        
        video_path = os.path.join("temp", uploaded_video.name)
        
        with open(video_path, "wb") as f:
            f.write(uploaded_video.read())
        
        st.video(video_path)
        
        if st.button("▶️ Process Video"):
            with st.spinner("Processing video..."):
                process_video(video_path)
            os.remove(video_path)
            
elif file_type == "Image URL":
    image_url = st.text_input("Enter the image URL:")
    if image_url:
        try:
            disease_result = get_disease_from_image_url(image_url)
            st.markdown('<div class="prediction-section">', unsafe_allow_html=True)
            st.markdown("### 🎯 Results")
            st.info(disease_result)
            st.markdown('</div>', unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error: {str(e)}")            

# Footer
st.markdown('<div class="footer">', unsafe_allow_html=True)
st.markdown(f"© {datetime.now().year} Plant Disease Classifier - AI-C-4 Team")
st.markdown('</div>', unsafe_allow_html=True)

num_classes = len(class_indices)

# Get list of unique plants (removing disease variations)
def get_unique_plants():
    plants = set()
    for class_name in class_indices.values():
        # Assuming class names are in format "plant_disease" or "plant_healthy"
        plant = class_name.split('_')[0]
        plants.add(plant)
    return sorted(list(plants))

unique_plants = get_unique_plants()

# Group diseases by plant
def group_diseases_by_plant():
    plant_diseases = {}
    for class_name in class_indices.values():
        parts = class_name.split('_')
        plant = parts[0]
        disease = '_'.join(parts[1:]) if len(parts) > 1 else 'healthy'
        
        if plant not in plant_diseases:
            plant_diseases[plant] = []
        plant_diseases[plant].append(disease)
    
    return plant_diseases

plant_diseases = group_diseases_by_plant()


# Add a new section for displaying supported plants and diseases
st.markdown('<div class="project-info">', unsafe_allow_html=True)
st.markdown("### 🌱 Supported Plants and Diseases")

# Create tabs for different views
tab1, tab2 = st.tabs(["Plants Overview", "Detailed Disease List"])

with tab1:
    # Display plants in a grid
    cols = st.columns(3)
    for idx, plant in enumerate(unique_plants):
        with cols[idx % 3]:
            st.markdown(f"#### {plant.title()}")
            st.write(f"Number of conditions: {len(plant_diseases[plant])}")

with tab2:
    # Display detailed disease list for each plant
    for plant in unique_plants:
        with st.expander(f"{plant.title()} Diseases"):
            for disease in plant_diseases[plant]:
                if disease == "healthy":
                    st.write("✅ Healthy")
                else:
                    st.write(f"🔍 {disease.replace('_', ' ').title()}")

st.markdown('</div>', unsafe_allow_html=True)


def get_disease_from_image_url(image_url):
    """
    Processes an image from a URL and returns the detected disease or a message.

    Parameters:
        image_url (str): The URL of the image.

    Returns:
        str: The disease name or a message indicating the image is not a plant.
    """
    try:
        # Download the image
        response = requests.get(image_url)
        response.raise_for_status()  # Ensure the request was successful
        image = Image.open(BytesIO(response.content))
        image_np = np.array(image)

        # Check if it's a plant image
        if not is_plant_image(image_np):
            return "The provided image is not of a plant."

        # Preprocess the image
        image_tensor = load_and_preprocess_image(image_np)

        # Perform prediction
        model.eval()
        with torch.no_grad():
            outputs = model(image_tensor)
            _, predicted = torch.max(outputs, 1)
            class_id = predicted.item()
            class_name = idx_to_class[class_id]
        
        return f"The detected plant disease is: {class_name}"
    except Exception as e:
        return f"Error processing the image: {str(e)}"