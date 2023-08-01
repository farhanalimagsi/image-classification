import tensorflow as tf
import tensorflow_hub as hub
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image


st.set_option('deprecation.showfileUploaderEncoding', False)
#@st.cache_resource()
#def load_model():
   # model = tf.keras.models.load_model('model.h5')
   # return model
@st.cache_resource()
def load_model_dog_breed():
    model_dog_breed = tf.keras.models.load_model('dog_breed.h5',
                                                custom_objects={"KerasLayer":hub.KerasLayer})
    return model_dog_breed


#def predict_class(image, model):
   # image = tf.cast(image, tf.float32)
    #image = tf.image.resize(image, [224, 224])
    #image = tf.expand_dims(image, axis=0)
    #prediction = model.predict(image)
    #return prediction

def predict_dog_class(image, model):
    imag = tf.cast(image, tf.float32)
    imag = tf.image.resize(imag, [224, 224])
    imag = tf.expand_dims(imag, axis=0)
    imag = imag/255
    dog_prediction = model.predict(imag)
    return dog_prediction

st.title("Multi-Class Image Classification")
st.sidebar.title("Select the Image Class")

app_mode = st.sidebar.radio("Pick image class", ["Dog Breed","Food"])

if app_mode == 'Dog Breed':
    dog_breed_model = load_model_dog_breed()
if app_mode == 'Food':  
    st.title("Under Process")

#if app_mode == 'Food':

    #file = st.file_uploader("Upload an image of food", type=["jpg", "jpeg", "png"])
   # col1, col2 = st.columns(2)

    #col1.subheader("Food Image Prediction")
    #col2.subheader("Result")

    #if file is None:
        #st.text("Waiting for upload....")
    #else: 
        #test_image = Image.open(file)
        #slot = st.empty()
        #col1.image(test_image, caption="Input Image", width=300)
        #slot.text('Running inference.....')
        #pred = tf.squeeze(predict_class(test_image, model))
        #class_names = pd.read_csv("class_names.txt")
        #class_names = np.array(class_names["labels"])
        #result = class_names[tf.argmax(pred)]
        #slot.text('Done')
        #output = 'The image is a ' + result
        #col2.success(output)


if app_mode == 'Dog Breed':
    
    st.subheader("Dog Image Classification")

    file = st.file_uploader("Upload an image of dog", type=["jpg", "jpeg", "png"])
    col1, col2 = st.columns(2)

    col1.subheader("dog Image Prediction")
    col2.subheader("Result")
    if file is None:
        st.text("Waiting for upload....")
    else: 
        test_image = Image.open(file)
        slot = st.empty()
        col1.image(test_image, caption="Input Image", width=300)
        slot.text('Running inference.....')
        pred_dog = tf.squeeze(predict_dog_class(test_image, dog_breed_model))
        unique_labels = pd.read_csv("dog_classes.txt")
        unique_labels = np.array(unique_labels["labels"])
        predicted_label = unique_labels[np.argmax(pred_dog)]
        slot.text('Done')
        output = 'The image is a ' + predicted_label
        col2.success(output)

