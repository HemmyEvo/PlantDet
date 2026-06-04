import streamlit as st
import tensorflow as tf
import numpy as np
import json
from PIL import Image

# --- 1. UI Styling & Configuration ---
st.set_page_config(page_title="Plant Disease AI", layout="centered")

st.markdown("""
    <style>
    .stApp { background-color: #0E1117; color: #FFFFFF; }
    .stButton>button {
        border: 2px solid #FFFFFF;
        border-radius: 4px;
        background-color: transparent;
        color: #FFFFFF;
        font-weight: bold;
        transition: 0.3s;
    }
    .stButton>button:hover { background-color: #FFFFFF; color: #000000; }
    </style>
    """, unsafe_allow_html=True)

st.title("🌿 Plant Disease AI")
st.write("Upload an image of a leaf to instantly identify potential diseases.")

# --- 2. Load the Master JSON Data ---
@st.cache_data
def load_class_data():
    try:
        with open('class_names.json', 'r') as f:
            classes = json.load(f)
            # Convert string number keys ("0") back to integers (0)
            return {int(k): v for k, v in classes.items()}
    except FileNotFoundError:
        st.error("Could not find 'class_names.json'. Please ensure it's in the same folder.")
        return {}

CLASS_DATA = load_class_data()

# --- 3. Load the Model ---
@st.cache_resource
def load_custom_model():
    return tf.keras.models.load_model('model/plant_det_model.keras')

model = load_custom_model()

# --- 4. File Uploader & Processing ---
uploaded_file = st.file_uploader("Select a leaf image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Leaf", use_container_width=True)
    
    with st.spinner("Analyzing leaf patterns..."):
        image = image.convert('RGB')
        image = image.resize((224, 224))
        
        img_array = np.array(image)
        img_array = np.expand_dims(img_array, axis=0)
        
        # --- 5. Prediction ---
        predictions = model.predict(img_array)
        predicted_index = int(np.argmax(predictions, axis=1)[0])
        
        # --- 6. Display Results ---
        st.write("---")
        st.subheader("Result:")
        
        # Look up the dictionary entry using the predicted number (e.g., 24)
        if predicted_index in CLASS_DATA:
            disease_info = CLASS_DATA[predicted_index]
            disease_name = disease_info["name"]
            
            # Format the title nicely
            clean_name = disease_name.replace("___", " - ").replace("_", " ")
            st.success(f"**{clean_name}**")
            
            # Show Cause and Remedy
            if "healthy" in disease_name.lower():
                st.balloons()
                st.info(f"**Advice:** {disease_info['remedy']}")
            else:
                st.warning(f"**Likely Cause:** {disease_info['cause']}")
                st.info(f"**Remedy:** {disease_info['remedy']}")
                
        else:
            st.error(f"Unknown Class Index Predicted: {predicted_index}")