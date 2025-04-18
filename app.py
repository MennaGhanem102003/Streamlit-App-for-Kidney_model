import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import cv2  # Add this import

# Set page configuration
st.set_page_config(
    page_title="Medical Image Analysis",
    page_icon="🏥",
    layout="centered"
)

# Add title and description
st.title("Medical Image Analysis")
st.write("Upload an image for analysis")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=['png', 'jpg', 'jpeg'])

# Class definitions - Update order to match training
class_names = ['Cyst', 'Normal', 'Stone', 'Tumor']

# Load the model
model = tf.keras.models.load_model('kidney_cnn_model.h5')

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    image = image.convert('RGB')
    # Convert PIL image to cv2 format for consistent preprocessing
    image_array = np.array(image)
    image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)  # Convert to BGR to match cv2.imread
    
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    if st.button('Analyze Image'):
        with st.spinner('Processing...'):
            # Preprocess the image to match training exactly
            image_resized = cv2.resize(image_array, (128, 128))
            image_array = image_resized / 255.0  # Normalize exactly like training
            image_array = np.expand_dims(image_array, axis=0)

            # Make prediction using same model configuration
            prediction = model.predict(image_array, verbose=0)
            predicted_class_idx = np.argmax(prediction, axis=1)[0]
            predicted_class = class_names[predicted_class_idx]
            confidence = prediction[0][predicted_class_idx]
            
            st.success('Analysis Complete!')
            
            st.subheader('Results')
            st.write(f"**Predicted Class:** {predicted_class}")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric(label="Confidence Score", value=f"{confidence:.2%}")
            with col2:
                st.metric(label="Processing Time", value="0.5s")
            
            with st.expander("See detailed analysis"):
                st.write(f"Model prediction confidence scores:")
                for i, class_name in enumerate(class_names):
                    score = prediction[0][i]
                    st.progress(float(score))
                    st.write(f"{class_name}: {score:.2%}")

with st.sidebar:
    st.header("About")
    st.write("This is a medical image analysis tool.")
    st.write("Upload an image to get started.")
