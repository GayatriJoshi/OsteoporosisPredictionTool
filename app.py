import streamlit as st
from PIL import Image
from keras.preprocessing import image
import numpy as np
import tensorflow as tf
import pandas as pd
from keras.models import load_model
from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from modelBMD import accuracy_of_model, conf_matrix, class_report
import modelBMD


#UI Variables
navigation_menus=["Prediction Based on BMD","Prediction based on X-ray"]
error_msg = f"""Oops, something went wrong."""
page_title="Osteoporosis Prediction Tool"
osteo_information="""
                   Osteoporosis is a bone condition with a slower bone formation rate than its breakdown. This leads to a degradation of Bone Mineral Density which in turn results in the deterioration of bone strength. Early prediction of this condition is not yet a consideration in the medical field because the visible effects of the disease are only seen at a crucial stage where it is guaranteed that the patient is osteoporotic. These visible effects include major fractures due to a small jerk or constant pain at certain bone joints and locations. This survey deals with analyzing the hotspot areas prone to the condition and the various technologies involved in the study and pre-diagnosis based on these hotspots along with various other factors. The most affected regions are the femoral neck, spinal region, wrist and dental area. The factors affecting the severity of the condition are age, gender, ethnicity, bone mineral density etc. 
            """
instructions="""
    This tool is used for Osteoporosis prediction by providing the following information. Enter the asked information and click on predict button.
    No personal data is stored here.
        """


@st.cache_data(show_spinner=False)
def load_banner():
    try:
        img=Image.open('Banner.png')
        st.image(img)
    except:
        pass

model = load_model('xray_classification_model.h5')
def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array / 255.0

def main():
    choice = st.sidebar.selectbox("Menu",navigation_menus)
    if choice == "Prediction Based on BMD":
        st.title(page_title)
    
        load_banner()
        st.markdown("""
                Here is an easy to use application for Osteoporosis Prediction.
                Following is some information regarding the condition:
            """)
        with st.expander("What is Osteoporosis?"):
            st.markdown(osteo_information)
        with st.expander("App Usage"):
            st.markdown(instructions)
        
        st.subheader('Enter the following details')
    
        age = st.number_input("Enter your age", min_value=18, max_value=100, value=40)
        bmd = st.number_input("Enter your Bone Mineral Density (BMD)", min_value=0.0, value=2.0, step=0.1)
        gender = st.radio("Select your gender", ["Male", "Female"])
        fracture_status = st.radio("Do you posses any fracture?",["Yes","No"])
    
        if st.button("Predict"):
            sex = 0
            if gender=="Male":
                sex=1
            else:
                sex=0
            fracture = 0
            if fracture_status=="Yes":
                fracture=1
            else:
                fracture = 0
            input_data = modelBMD.standard.transform([[sex,age,fracture,bmd]])
            prediction = modelBMD.model.predict(input_data)
        
            if prediction[0]==1:
                st.subheader('Person is Osteoporotic')
            else:
                st.subheader('Person is not Osteoporotic')
            st.write("Accuracy:",accuracy_of_model)
            st.write("Confusion Matrix:",conf_matrix)
            st.text(class_report)

    elif choice=="Prediction based on X-ray":
        st.title("Prediction based on X-ray")
        uploaded_file = st.file_uploader("Choose a file", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            img_array = preprocess_image(uploaded_file)
            prediction = model.predict(img_array)
            if prediction[0][0] > 0.6:
                st.success(f"Prediction: Osteoporotic")
            else:
                st.success(f"Prediction: Non - Osteoporotic")
if __name__ == '__main__':
    main()