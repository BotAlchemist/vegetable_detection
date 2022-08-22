# -*- coding: utf-8 -*-
"""
Created on Fri Aug 19 18:56:55 2022

@author: Sumit
"""

import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import glob


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

# load model
loaded_model= load_model(filename)

tab1, tab2= st.tabs(['Upload Image', 'Open Camera'])
with tab1:
    img_file_buffer = st.file_uploader("Upload Images", type=["png","jpg","jpeg"])
    image= Image.open(img_file_buffer)
    
    if img_file_buffer is not None:
        st.image(image,width=200)
       
        
        
        #img= img.resize((IMAGE_SIZE*IMAGE_SIZE))#
        image.save("Unseen/test.jpg")
        
        all_files= glob.glob("Unseen/*")
        
        X_unseen=[]
        for i_ in all_files:
            image=  tf.keras.preprocessing.image.load_img(
                path= i_,
                target_size=(IMAGE_SIZE,IMAGE_SIZE),
                )
            
       
        image= np.array(image)
        img_batch = np.expand_dims(image, 0)
        prediction= loaded_model.predict(img_batch)
        prediction= prediction[0]
        index= np.argmax(prediction)
        predicted_label= CLASS_NAMES[index]
        predicted_confidence= prediction[index]
        st.subheader(CLASS_NAMES[index])
        st.write("Confidence: ", predicted_confidence, " %")
with tab2:
    img_file_buffer = st.camera_input("Take a picture")


    
    
    


