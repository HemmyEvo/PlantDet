import streamlit as st
import tensorflow as tf
import numpy as np
import json
from PIL import Image

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

@st.cache_data
def load_class_names():
    try:
        with open('class_names.json', 'r') as f:
            classes = json.load(f)
            # JSON keys are always strings, convert them back to integers
            return {int(k): v for k, v in classes.items()}
    except FileNotFoundError:
        st.error("Could not find 'class_names.json'. Please ensure it's in the same folder.")
        return {}

CLASS_NAMES = load_class_names()

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
        
        # Get the string name from our loaded JSON dictionary
        disease_name = CLASS_NAMES.get(predicted_index, f"Unknown Class Index: {predicted_index}")
        
        # --- 6. Display Results ---
        st.write("---")
        st.subheader("Result:")
        
        clean_name = disease_name.replace("___", " - ").replace("_", " ")
        st.success(f"**{clean_name}**")