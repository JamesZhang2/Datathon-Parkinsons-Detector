import streamlit as st
from model import ParkinsonPredictor

# import pandas as pd
# import numpy as np
from PIL import Image

# import base64

st.set_page_config(layout="wide")

st.markdown(
    """
<style>
.big-font {
    font-size:100px !important;
    font-family:Trebuchet MS;
    font-color:#63534f;
    
}
.med-font {
    font-size:50px !important;
    font-family:Trebuchet MS;
}
.small-font {
    font-size:35px !important;
    font-family:Roboto;
}
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div style=
        'background-color: #f5d0c6; 
        padding: 20px;
        text-align: center;
        '>
    <p class="big-font">Do you have Parkinson\'s?</p>
    
    </div>
    
    """,
    unsafe_allow_html=True,
)


# st.markdown('<p class="big-font">Do you have Parkinson\'s?</p>', unsafe_allow_html=True)

# Titling
st.markdown(
    """
     <div style=
        'background-color: #FFFFFF; 
        padding: 10px;
        text-align: center;
        '>
    <p class="med-font">Draw some images to find out!✨📝</p>
    """,
    unsafe_allow_html=True,
)
st.markdown(
    '<p class="small-font">Download the image template given below...</p>',
    unsafe_allow_html=True,
)


# st.title("Download the image template given below!")


# Define some colors to use in the app
primary_color = "#EE4266"
secondary_color = "#EFE9F4"
text_color = "#2B2D42"

with st.container():
    st.markdown(
        f'<p style="color: {text_color};">Step-by-step guidelines:</p>',
        unsafe_allow_html=True,
    )

st.title("Image Uploader")

# Create a file uploader widget that accepts image files
uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])


@st.cache_data
def init_model(num_epochs):
    print("initializing model")
    return ParkinsonPredictor(num_epochs)


p_cnn = init_model(40)

# If an image file was uploaded, display it
if uploaded_file is not None:
    # Open the uploaded image using PIL
    image = Image.open(uploaded_file)
    prediction = p_cnn.predict(image)
    # Display the image in the app
    st.image(image, caption="Uploaded Image", use_column_width=True)
    if prediction == 0:
        print("Healthy")
        st.write("You do not have Parkinson's")
    else:
        print("Patient")
        st.write("You have Parkinson's")
