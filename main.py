import streamlit as st
import tensorflow as tf
import numpy as np


#Tensorflow Model Prediction
def model_prediction(test_image):
    model = tf.keras.models.load_model("trained_plant_disease_model.keras")
    image = tf.keras.preprocessing.image.load_img(test_image,target_size=(128,128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr]) #convert single image to batch
    predictions = model.predict(input_arr)
    return np.argmax(predictions) #return index of max element

#Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page",["Home","About","Prediction"])

#Main Page
if(app_mode=="Home"):
    st.header("PLANT DISEASE RECOGNITION SYSTEM")
    image_path = "/content/drive/MyDrive/MyPrograms/home_page.jpg"
    st.image(image_path,use_column_width=True)
    st.markdown("""#Plant Disease Recognition System 🌿🔍

    **Introduction**

    Welcome to the Plant Disease Recognition System, a cutting-edge project developed for our final year project. Our system aims to assist in the efficient identification of plant diseases. By uploading an image of a plant, our system can quickly analyze it to detect any signs of diseases, helping to protect crops and ensure a healthier harvest.

    ### How It Works
    1. **Upload Image:** Navigate to the **Prediction** page and upload an image of a plant showing symptoms of disease.
      
    2. **Analysis:** Our system utilizes state-of-the-art machine learning algorithms to process the image and identify potential diseases.
      
    3. **Results:** View detailed results and receive recommendations for further action.

    ### Features
    - **Accuracy:** Our system employs advanced machine learning techniques for accurate disease detection.
      
    - **User-Friendly:** Enjoy a simple and intuitive interface designed for a seamless user experience.
      
    - **Fast and Efficient:** Receive results in seconds, enabling quick decision-making in the field.

    ### Dataset
    Our system is trained on a comprehensive dataset of about 87,000 RGB images of healthy and diseased crop leaves, categorized into 38 different classes. The dataset is divided into an 80/20 ratio of training and validation sets, preserving the directory structure. Additionally, a new directory containing 33 test images has been created for prediction purposes.

    ### Get Started
    Click on the **Prediction** page in the sidebar to upload an image and experience the power of our Plant Disease Recognition System!

    ### About Us
    Learn more about our project, team members, and goals on the **About** page."""
  )

#About Project
elif(app_mode=="About"):
    st.header("About")
    image_path = "/content/drive/MyDrive/MyPrograms/about_img.jpg"
    st.markdown("""
                ### About the Dataset

                This dataset has been created through offline augmentation from the original dataset, comprising approximately 87,000 RGB images of both healthy and diseased crop leaves categorized into 38 different classes. The dataset is structured into training and validation sets with an 80/20 split, preserving the directory structure.

                #### Dataset Contents
                - **Train Dataset:** 70,295 images
                - **Test Dataset:** 33 images
                - **Validation Dataset:** 17,572 images

                 ### Team Members:
                - Aditi Tomar (Roll No. 12345)
                - Ansh Agarwal (Roll No. 23456)
                - Atul (Roll No. 34567)
                - Ankit Bisht(Roll No. 45678)
                """)

#Prediction Page
elif(app_mode=="Prediction"):
    st.header("Prediction")
    test_image = st.file_uploader("Choose an Image:")
    if(st.button("Show Image")):
        st.image(test_image,width=4,use_column_width=True)
    #Predict button
    if(st.button("Predict")):
        st.write("Our Prediction")
        result_index = model_prediction(test_image)
        #Reading Labels
        class_name = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
                    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 
                    'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 
                    'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 
                    'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 
                    'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
                    'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 
                    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 
                    'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 
                    'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 
                    'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 
                    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 
                    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
                      'Tomato___healthy']
        st.success("Model is Predicting it's a {}".format(class_name[result_index]))