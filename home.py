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
    font-size:80px !important;
    font-family:Trebuchet MS;
    color:#63534f;
}
.med-font {
    font-size:50px !important;
    font-family:Trebuchet MS;
}
.small-font {
    font-size:35px !important;
    font-family:Roboto;
}
.green-text {
    font-size:30px !important;
    font-family:Trebuchet MS;
    color: #008000;
}
.red-text {
    font-size:30px !important;
    font-family:Trebuchet MS;
    color: #C00000;
}
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div style=
        'background-color: #e0f1ff; 
        padding:10px;
        margin-top: -40px;
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
        border-radius: 10px;
        '>
    <p class="med-font">Draw some images to find out!✨📝</p>
    """,
    unsafe_allow_html=True,
)

# Split into columns

left_col, right_col = st.columns(2, gap="large")

# left_col.markdown(
#     '<p class="small-font">Download the image template given below...</p>',
#     unsafe_allow_html=True,
# )


left_col.header("Download the image template below!")


# Define some colors to use in the app
primary_color = "#EE4266"
secondary_color = "#EFE9F4"
text_color = "#2B2D42"

# TODO: Add download button

left_col.markdown(
    f"""
    <p style="color: {text_color};">Step-by-step guidelines:</p>
    <ol>
    <li>Download the image template
    <li>Print the image out
    <li>Trace the black line with a blue pen
    <li>Take a good picture of the drawing, with correct orientation and cropping
    <li>Upload the image and see the result!
    </ol>
    </br>
    </br>
    """,
    unsafe_allow_html=True,
)

with left_col:
    l_left_col, l_right_col = st.columns(2, gap="small")

template = Image.open("meanderTemplate.jpg")

# Display the template in the app
l_left_col.write("Template:")
l_left_col.image(template, width=200)

# file_path = "meanderTemplate.png"
# button = st.download_button(label="Download Template", data=file_path)

with open("meanderTemplate.jpg", "rb") as image:
    l_left_col.download_button(
        "Download Template", data=image, file_name="template.jpg"
    )

l_right_col.write("Example:")
example = Image.open("example.jpg")
l_right_col.image(example, width=200)

# Right column

right_col.header("Image Uploader")

# Create a file uploader widget that accepts image files
uploaded_file = right_col.file_uploader(
    "Choose an image file", type=["jpg", "jpeg", "png"]
)


@st.cache_data
def init_model(num_epochs):
    print("Initializing model")
    return ParkinsonPredictor(num_epochs)


p_cnn = init_model(40)

# If an image file was uploaded, display it
if uploaded_file is not None:
    # Open the uploaded image using PIL
    image = Image.open(uploaded_file)
    # print(image.mode)
    if image.mode != "RGB":
        image = image.convert("RGB")
    prediction = p_cnn.predict(image)
    # Display the image in the app
    right_col.image(image, caption="Uploaded Image", width=300)

    # left_col
    if prediction == 0:
        print("Healthy")
        right_col.markdown(
            """<p class="green-text">You probably do not have Parkinson's.</p>""",
            unsafe_allow_html=True,
        )
    else:
        right_col.markdown(
            """<p class="red-text">You may have Parkinson's. Go visit a doctor.</p>""",
            unsafe_allow_html=True,
        )

    right_col.write(
        "Disclaimer: Please note that this model is intended for informational purposes only and may not be accurate or up-to-date. The model is not intended to replace professional medical advice, diagnosis, or treatment. Always consult a qualified healthcare provider for any questions you may have regarding a medical condition. Never disregard professional medical advice or delay in seeking it because of this medical model."
    )
