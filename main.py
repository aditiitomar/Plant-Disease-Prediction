import os
import json
from PIL import Image

import numpy as np
import tensorflow as tf
import streamlit as st


working_dir = os.path.dirname(os.path.abspath(__file__))
model_path = f"{working_dir}/model.h5"
# Load the pre-trained model
model = tf.keras.models.load_model(model_path)

# loading the class names
class_indices = json.load(open(f"{working_dir}/class_indices.json"))


# Function to Load and Preprocess the Image using Pillow
def load_and_preprocess_image(image_path, target_size=(224, 224)):
    # Load the image
    img = Image.open(image_path)
    # Resize the image
    img = img.resize(target_size)
    # Convert the image to a numpy array
    img_array = np.array(img)
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    # Scale the image values to [0, 1]
    img_array = img_array.astype('float32') / 255.
    return img_array


# Function to Predict the Class of an Image
def predict_image_class(model, image_path, class_indices):
    preprocessed_img = load_and_preprocess_image(image_path)
    predictions = model.predict(preprocessed_img)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class_name = class_indices[str(predicted_class_index)]
    return predicted_class_name


# Streamlit App

st.header("Plant Disease Recognition System 🌿🔍")
image_path = "home_page.jpg"
st.image(image_path, use_column_width=True)
st.markdown(""" 
### Introduction
Welcome to the Plant Disease Recognition System, a cutting-edge project developed for our final year project. Our system aims to assist in the efficient identification of plant diseases. By uploading an image of a plant, our system can quickly analyze it to detect any signs of diseases, helping to protect crops and ensure a healthier harvest.
     
### Team Members
- Aditi Tomar (Roll No. 12345)
- Ansh Agarwal (Roll No. 23456)
- Atul (Roll No. 34567)
- Ankit Bisht (Roll No. 45678)
    
### How It Works
- **Upload Image:** Navigate to the **Prediction** page and upload an image of a plant showing symptoms of disease.
- **Analysis:** Our system utilizes state-of-the-art machine learning algorithms to process the image and identify potential diseases.
- **Results:** View detailed results and receive recommendations for further action.

### Features
- **Accuracy:** Our system employs advanced machine learning techniques for accurate disease detection.
- **User-Friendly:** Enjoy a simple and intuitive interface designed for a seamless user experience.
- **Fast and Efficient:** Receive results in seconds, enabling quick decision-making in the field.

### Dataset    
This dataset is part of a crowdsourcing effort to leverage smartphone technology and machine learning for improving food production by addressing the issue of infectious diseases in crops.
- Size: Over 50,000 expertly curated images.
- Content: The images consist of healthy and infected leaves of crop plants.
- Platform: The dataset is released on the PlantVillage online platform.
- Purpose: The dataset is intended to facilitate the development of mobile disease diagnostics using machine learning and crowdsourcing.
- Goal: The goal is to use computer vision approaches to address yield losses in crop plants caused by infectious diseases.

### Prediction
Upload an image and experience the power of our Plant Disease Recognition System!"""
)

uploaded_image = st.file_uploader("", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    col1, col2 = st.columns(2)

    with col1:
        resized_img = image.resize((150, 150))
        st.image(resized_img)

    with col2:
        if st.button('Classify'):
            # Preprocess the uploaded image and predict the class
            prediction = predict_image_class(model, uploaded_image, class_indices)
            st.success(f'Prediction: {str(prediction)}')
