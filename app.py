import streamlit as st
import tensorflow as tf
import numpy as np
import json
from PIL import Image

# --- 1. UI Styling & Configuration ---
# Setting layout to 'wide' gives us more room to make things look clean
st.set_page_config(page_title="Plant Disease AI", layout="wide", page_icon="🌿")

# Polished CSS for a premium feel
st.markdown("""
    <style>
    .stApp { background-color: #0E1117; color: #FFFFFF; }
    
    /* Style the file uploader box */
    [data-testid="stFileUploadDropzone"] {
        border: 2px dashed #4CAF50;
        border-radius: 10px;
        background-color: rgba(76, 175, 80, 0.05);
    }
    
    /* Make the success text stand out */
    .big-font {
        font-size: 24px !important;
        font-weight: bold;
        color: #4CAF50;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 2. Load the Master JSON Data ---
@st.cache_data
def load_class_data():
    try:
        with open('class_names.json', 'r') as f:
            classes = json.load(f)
            return {int(k): v for k, v in classes.items()}
    except FileNotFoundError:
        st.error("Could not find 'class_names.json'. Please ensure it's in the same folder.")
        return {}

CLASS_DATA = load_class_data()

# --- 3. Extract Unique Plants for the Sidebar ---
def get_supported_plants(data):
    plants = set()
    for key, val in data.items():
        # Grab just the plant name (e.g., "Apple" from "Apple___Black_rot")
        plant_name = val["name"].split("___")[0].replace("_", " ")
        plants.add(plant_name)
    return sorted(list(plants))

# --- 4. Sidebar UI ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/628/628283.png", width=80) # Cute little plant icon
    st.title("Supported Plants")
    st.write("This AI is trained to detect diseases in the following leaves:")
    
    # Display the plants dynamically as a clean list
    supported_plants = get_supported_plants(CLASS_DATA)
    for plant in supported_plants:
        st.markdown(f"- **{plant}**")
        
    st.markdown("---")
    st.info("💡 Tip: Make sure the leaf is clearly visible and well-lit in your photo for the best results.")

# --- 5. Main Page UI ---
st.title("🌿 Plant Disease AI Diagnostic Tool")
st.write("Upload a clear photo of a leaf to instantly identify potential diseases and get actionable treatment remedies.")
st.markdown("---")

# --- 6. Load the Model ---
@st.cache_resource
def load_custom_model():
    return tf.keras.models.load_model('model/plant_det_model.keras')

model = load_custom_model()

# --- 7. File Uploader & Processing ---
uploaded_file = st.file_uploader("Drop a leaf image here or click to browse...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Use columns to center the image nicely instead of it taking up the whole screen
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        image = Image.open(uploaded_file)
        # Display the image with rounded corners via Streamlit's built-in styling
        st.image(image, caption="Your Uploaded Leaf", use_container_width=True)
        
        with st.spinner("🔬 Analyzing leaf cellular patterns..."):
            # Processing
            image = image.convert('RGB')
            image = image.resize((224, 224))
            img_array = np.array(image)
            img_array = np.expand_dims(img_array, axis=0)
            
            # Prediction
            predictions = model.predict(img_array)
            predicted_index = int(np.argmax(predictions, axis=1)[0])
            
            st.markdown("---")
            
            # --- 8. Display Results ---
            if predicted_index in CLASS_DATA:
                disease_info = CLASS_DATA[predicted_index]
                disease_name = disease_info["name"]
                
                # Format the name nicely
                clean_name = disease_name.replace("___", " - ").replace("_", " ")
                
                st.markdown(f"### Diagnosis: <span class='big-font'>{clean_name}</span>", unsafe_allow_html=True)
                
                # Logic for Healthy vs Diseased
                if "healthy" in disease_name.lower():
                    st.balloons()
                    st.success("🎉 Good news! This plant looks perfectly healthy.")
                    st.info(f"**Maintenance Advice:** {disease_info['remedy']}")
                else:
                    st.warning(f"**🦠 Likely Cause:** {disease_info['cause']}")
                    st.error(f"**🛠️ Remedy:** {disease_info['remedy']}")
                    
            else:
                st.error(f"Unknown Class Index Predicted: {predicted_index}")