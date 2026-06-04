import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# --- 1. UI Styling & Configuration ---
st.set_page_config(page_title="Plant Disease AI", layout="centered")

# Injecting minimal, high-contrast CSS
st.markdown("""
    <style>
    .stApp {
        background-color: #0E1117;
        color: #FFFFFF;
    }
    .stButton>button {
        border: 2px solid #FFFFFF;
        border-radius: 4px;
        background-color: transparent;
        color: #FFFFFF;
        font-weight: bold;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #FFFFFF;
        color: #000000;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🌿 Plant Disease AI")
st.write("Upload an image of a leaf to instantly identify potential diseases.")

# --- 2. The Class Names Dictionary ---
# CRITICAL: You must copy your EXACT list of class names from your Jupyter Notebook
# The model outputs a number (like 24), and this list translates it to a string.
CLASS_NAMES = [
    # Replace these with your actual 38 classes in the exact order!
    "Apple___Apple_scab", "Apple___Black_rot", "Apple___Cedar_apple_rust", "Apple___healthy", 
    "Blueberry___healthy", "Cherry_(including_sour)___Powdery_mildew", "Cherry_(including_sour)___healthy", 
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot", "Corn_(maize)___Common_rust_", 
    "Corn_(maize)___Northern_Leaf_Blight", "Corn_(maize)___healthy", "Grape___Black_rot", 
    "Grape___Esca_(Black_Measles)", "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)", "Grape___healthy", 
    "Orange___Haunglongbing_(Citrus_greening)", "Peach___Bacterial_spot", "Peach___healthy", 
    "Pepper,_bell___Bacterial_spot", "Pepper,_bell___healthy", "Potato___Early_blight", 
    "Potato___Late_blight", "Potato___healthy", "Raspberry___healthy", "Soybean___healthy", 
    "Squash___Powdery_mildew", "Strawberry___Leaf_scorch", "Strawberry___healthy", 
    "Tomato___Bacterial_spot", "Tomato___Early_blight", "Tomato___Late_blight", "Tomato___Leaf_Mold", 
    "Tomato___Septoria_leaf_spot", "Tomato___Spider_mites Two-spotted_spider_mite", "Tomato___Target_Spot", 
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus", "Tomato___Tomato_mosaic_virus", "Tomato___healthy"
]

# --- 3. Load the Model ---
# @st.cache_resource ensures the model only loads once, preventing the app from freezing on every click.
@st.cache_resource
def load_custom_model():
    # Make sure this filename perfectly matches the model saved on your computer
    return tf.keras.models.load_model('model/plant_det_model.keras')

model = load_custom_model()

# --- 4. File Uploader & Processing ---
uploaded_file = st.file_uploader("Select a leaf image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Open and display the image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Leaf", use_container_width=True)
    
    with st.spinner("Analyzing leaf patterns..."):
        # Convert image to RGB (removes transparent layers if it's a PNG)
        image = image.convert('RGB')
        
        # Resize to exactly 224x224 (MobileNetV2's requirement)
        image = image.resize((224, 224))
        
        # Convert to numpy array
        img_array = np.array(image)
        
        # Expand dimensions from (224, 224, 3) to (1, 224, 224, 3) so the model knows it's 1 batch
        img_array = np.expand_dims(img_array, axis=0)
        
        # Apply standard normalization (if you used this during training)
        # img_array = img_array / 255.0  
        
        # --- 5. Prediction ---
        predictions = model.predict(img_array)
        
        # tf.argmax gets the highest probability number (just like in your notebook)
        predicted_index = np.argmax(predictions, axis=1)[0]
        
        # Get the actual string name from our list
        if predicted_index < len(CLASS_NAMES):
            disease_name = CLASS_NAMES[predicted_index]
        else:
            disease_name = f"Unknown Class Index: {predicted_index}"
        
        # --- 6. Display Results ---
        st.write("---")
        st.subheader("Result:")
        
        # Clean up the output string (e.g., change "Soybean___healthy" to "Soybean - Healthy")
        clean_name = disease_name.replace("___", " - ").replace("_", " ")
        
        st.success(f"**{clean_name}**")