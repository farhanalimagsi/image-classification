import tensorflow as tf
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.metrics import classification_report


st.set_option('deprecation.showfileUploaderEncoding', False)
@st.cache_resource()
def load_model():
    model = tf.keras.models.load_model('model.h5')
    return model

def predict_class(image, model):
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, [224, 224])
    image = tf.expand_dims(image, axis=0)
    prediction = model.predict(image)
    return prediction

model = load_model()
st.title("Food Vision Classification")
file = st.file_uploader("Upload an iamge of food", type=["jpg", "jpeg", "png"])


col1, col2 = st.columns(2)

col1.subheader("Image to be Predicted")
col2.subheader("Results")
if file is None:
    st.text("Waiting for upload....")
else: 
    test_image = Image.open(file)
    slot = st.empty()
    col1.image(test_image, caption="Input Image", width=300)
    slot.text('Running inference.....')
    pred = tf.squeeze(predict_class(test_image, model))
    class_names = pd.read_csv("class_names.txt")
    class_names = np.array(class_names["labels"])
    result = class_names[tf.argmax(pred)]
    slot.text('Done')
    output = 'The image is a ' + result
    col2.success(output)

st.subheader("Metrics")  

