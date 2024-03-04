import streamlit as st
from PIL import Image
import numpy as np
from utils import *
from streamlit_lottie import st_lottie

lottie_file = load_lottieurl()  # animation url

st.set_page_config(page_title="im_col")

st_lottie(lottie_file, height=175, quality="medium")
st.title("Image Colourizer")

st.write(
    "Upload any black and white landscape image and colourize it using a deep learning Convolutional Autoencoder."
)

st.write(
    "*Disclaimer: uploaded images are not stored. Read below for more information about model training and performance.*"
)

if "uploaded_img" not in st.session_state:
    st.session_state["uploaded_img"] = None

if "processed_img" not in st.session_state:
    st.session_state["processed_img"] = None


uploaded_file = st.file_uploader(
    "Choose a JPG or PNG file",
    type=["jpg", "png"],
)

if uploaded_file is not None:
    st.session_state["uploaded_img"] = resize_image(Image.open(uploaded_file))

if st.button("❌"):
    # Reset the session state
    st.session_state["uploaded_img"] = None
    st.session_state["processed_img"] = None

if st.session_state["uploaded_img"] is not None:
    # Display the uploaded image
    image = st.session_state["uploaded_img"]

    with st.spinner("Colourizing image now..."):
        processed_img = colourize_image(image)
        if processed_img is False:
            st.session_state["processed_img"] = False
        else:
            st.session_state["processed_img"] = processed_img

if st.session_state["processed_img"] is False:
    st.error("Uh oh, an error occured. Please try again later.", icon="⛔")

elif st.session_state["processed_img"] is not None:
    col1, col2 = st.columns(2)

    with col1:
        st.write("**Uploaded**")
        st.image(st.session_state["uploaded_img"], use_column_width=True)

    with col2:
        st.write("**Prediction**")
        st.image(st.session_state["processed_img"], use_column_width=True)

st.divider()

st.write("### Model Training Set")

st.write(
    "Below are a few examples from the dataset that the model was trained on. As you can tell, the images are lower quality and so the performance was certainly impacted by data. Moreover, the images tend to have a 'blue bias', meaning that images largely contained a sky or water component that informed predictions heavily as well. However, with some relatively simple architecture and light training, our model performs respectably."
)

st.write(
    "You can read more about the model architecture and training process on my [GitHub repo](https://github.com/akashvshroff/ImageColourizer)."
)

col1, col2, col3 = st.columns(3)
imgs = [4, 14, 3118]
with col1:
    st.write("**Grayscale**")
    # Assuming you have images saved in the same directory as your app
    for img in imgs:
        st.image(f"images/gray_{img}.jpg", use_column_width=True)

with col2:
    st.write("**True Color**")
    for img in imgs:
        st.image(f"images/col_{img}.jpg", use_column_width=True)

with col3:
    st.write("**Prediction**")
    for img in imgs:
        st.image(f"images/pred_{img}.png", use_column_width=True)
