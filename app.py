# -*- coding: utf-8 -*-
"""
Created on Fri Aug 19 18:56:55 2022

@author: Sumit
"""

import streamlit as st
from streamlit_option_menu import option_menu
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np



img_favicon = Image.open('icons/favicon.JPG')


st.set_page_config(layout='wide',page_title="Veggies Detector", page_icon = img_favicon)
st.set_option('deprecation.showPyplotGlobalUse', False)

#st.beta_set_page_config(page_title='your_title', page_icon = favicon, layout = 'wide', initial_sidebar_state = 'auto')
# favicon being an object of the same kind as the one you should provide st.image() with (ie. a PIL array for example) or a string (url or local file path)
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 


IMAGE_SIZE=200
filename= 'vegetable_cnn_model.h5'

CLASS_NAMES= ['Bean',
 'Bitter_Gourd',
 'Bottle_Gourd',
 'Brinjal',
 'Broccoli',
 'Cabbage',
 'Capsicum',
 'Carrot',
 'Cauliflower',
 'Cucumber',
 'Papaya',
 'Potato',
 'Pumpkin',
 'Radish',
 'Tomato']



img_banner = Image.open('icons/icon.JPG')

st.sidebar.image(img_banner)

with st.sidebar:
    i_page= option_menu('Veggies Detector', ['Home',  'Detector'],
                        default_index=0, icons=['house', 'search' ], menu_icon= 'cast')

st.sidebar.markdown("##### Developed by: Sumit Srivastava")

if i_page == 'Home':
    st.markdown("## Introduction")
    st.write(''' 
             The objective of this app is to perform real time detection of vegetables.
             User can upload the picture of the vegetable or click the picture via their camera.
             
             Vegetable images were taken from Kaggle: https://www.kaggle.com/code/darkalchemist/cnn-model-vegetables/data.
             In this dataset there were 21000 images from 15 classes, where each class contains a total of 1400 images.
             ''')
    st.markdown("---")        
    st.markdown("##### Find code on GitHub: https://github.com/BotAlchemist/vegetable_detection")
    

elif i_page== 'Detector':
    # load model
    loaded_model= load_model(filename)
    
    tab1, tab2= st.tabs(['Upload Image', 'Open Camera'])
    with tab1:
        img_file_buffer = st.file_uploader("Upload Image", type=["png","jpg","jpeg"])
        
        
        if img_file_buffer is not None:
            image= Image.open(img_file_buffer)
            st.image(image,width=200)
           
            
            
            #img= img.resize((IMAGE_SIZE*IMAGE_SIZE))#
            image.save("Unseen/test.jpg")
            
            
            image=  tf.keras.preprocessing.image.load_img(
                    path= r'Unseen/test.jpg',
                    target_size=(IMAGE_SIZE,IMAGE_SIZE),
                    )
                
           
            image= np.array(image)
            img_batch = np.expand_dims(image, 0)
            prediction= loaded_model.predict(img_batch)
            prediction= prediction[0]
            index= np.argmax(prediction)
            predicted_label= CLASS_NAMES[index]
            predicted_confidence= prediction[index]
            st.subheader("Probably a " +  CLASS_NAMES[index])
            st.markdown("##### Confidence: "+ str(predicted_confidence*100) + " %")
    
    
    with tab2:
        cam_file_buffer = st.camera_input("Take a picture")
        if cam_file_buffer is not None:
            img = Image.open(cam_file_buffer)
            img.save("Unseen/cam.jpg")
            
            cam_image=  tf.keras.preprocessing.image.load_img(
                    path= r'Unseen/cam.jpg',
                    target_size=(IMAGE_SIZE,IMAGE_SIZE),
                    )
            
            image= np.array(cam_image)
            img_batch = np.expand_dims(image, 0)
            prediction= loaded_model.predict(img_batch)
            prediction= prediction[0]
            index= np.argmax(prediction)
            predicted_label= CLASS_NAMES[index]
            predicted_confidence= prediction[index]
            st.subheader("Probably a " +  CLASS_NAMES[index])
            st.markdown("##### Confidence: "+ str(predicted_confidence*100) + " %")


    
    
    


